# Copyright 2024 The coix Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example: Time Series Model - Bouncing MNIST in NumPyro
======================================================

This example illustrates how to construct an inference program based on the APGS
sampler [1] for BMNIST. The details of BMNIST can be found in the sections
6.4 and F.3 of the reference. We will use the NumPyro (default) backend for this
example.

**References**

    1. Wu, Hao, et al. Amortized population Gibbs samplers with neural
       sufficient statistics. ICML 2020.

.. image:: ../_static/bmnist.gif
    :align: center

"""

import argparse
from functools import partial

import coix
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

# %%
# First, let's load the moving mnist dataset.


def load_dataset(*, is_training, batch_size):
  ds = tfds.load("moving_mnist:1.0.0", split="test")
  ds = ds.repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
    map_fn = lambda x: x["image_sequence"][..., :10, :, :, 0] / 255
  else:
    map_fn = lambda x: x["image_sequence"][..., 0] / 255
  ds = ds.batch(batch_size)
  ds = ds.map(map_fn)
  return iter(tfds.as_numpy(ds))


def get_digit_mean():
  ds, ds_info = tfds.load("mnist:3.0.1", split="train", with_info=True)
  ds = tfds.as_numpy(ds.batch(ds_info.splits["train"].num_examples))
  digit_mean = next(iter(ds))["image"].squeeze(-1).mean(axis=0)
  return digit_mean / 255


# %%
# Next, we define the neural proposals for the Gibbs kernels and the neural
# decoder for the generative model.


def scale_and_translate(image, where, out_size):
  translate = abs(image.shape[-1] - out_size) * (where[..., ::-1] + 1) / 2
  return jax.image.scale_and_translate(
      image,
      (out_size, out_size),
      (0, 1),
      jnp.ones(2),
      translate,
      method="cubic",
      antialias=False,
  )


def crop_frames(frames, z_where, digit_size=28):
  # frames:           time.frame_size.frame_size
  # z_where: (digits).time.2
  # out:     (digits).time.digit_size.digit_size
  if frames.ndim == 2 and z_where.ndim == 1:
    return scale_and_translate(frames, z_where, out_size=digit_size)
  elif frames.ndim == 3 and z_where.ndim == 2:
    in_axes = (0, 0)
  elif frames.ndim == 3 and z_where.ndim == 3:
    in_axes = (None, 0)
  elif frames.ndim == z_where.ndim:
    in_axes = (0, 0)
  elif frames.ndim > z_where.ndim:
    in_axes = (0, None)
  else:
    in_axes = (None, 0)
  return jax.vmap(partial(crop_frames, digit_size=digit_size), in_axes)(
      frames, z_where
  )


def embed_digits(digits, z_where, frame_size=64):
  # digits:  (digits).      .digit_size.digit_size
  # z_where: (digits).(time).2
  # out:     (digits).(time).frame_size.frame_size
  if digits.ndim == 2 and z_where.ndim == 1:
    return scale_and_translate(digits, z_where, out_size=frame_size)
  elif digits.ndim == 2 and z_where.ndim == 2:
    in_axes = (None, 0)
  elif digits.ndim >= z_where.ndim:
    in_axes = (0, 0)
  else:
    in_axes = (None, 0)
  return jax.vmap(partial(embed_digits, frame_size=frame_size), in_axes)(
      digits, z_where
  )


def conv2d(frames, digits):
  # frames:          (time).frame_size.frame_size
  # digits: (digits).      .digit_size.digit_size
  # out:    (digits).(time).conv_size .conv_size
  if frames.ndim == 2 and digits.ndim == 2:
    return jax.scipy.signal.convolve2d(frames, digits, mode="valid")
  elif frames.ndim == digits.ndim:
    in_axes = (0, 0)
  elif frames.ndim > digits.ndim:
    in_axes = (0, None)
  else:
    in_axes = (None, 0)
  return jax.vmap(conv2d, in_axes=in_axes)(frames, digits)


class EncoderWhat(nn.Module):

  @nn.compact
  def __call__(self, digits):
    x = digits.reshape(digits.shape[:-2] + (-1,))
    x = nn.Dense(400)(x)
    x = nn.relu(x)
    x = nn.Dense(200)(x)
    x = nn.relu(x)

    x = x.sum(-2)  # sum/mean across time
    loc_raw = nn.Dense(10)(x)
    scale_raw = 0.5 * nn.Dense(10)(x)
    return loc_raw, jnp.exp(scale_raw)


class EncoderWhere(nn.Module):

  @nn.compact
  def __call__(self, frame_conv):
    x = frame_conv.reshape(frame_conv.shape[:-2] + (-1,))
    x = nn.softmax(x, -1)
    x = nn.Dense(200)(x)
    x = nn.relu(x)
    x = nn.Dense(200)(x)
    x = x.reshape(x.shape[:-1] + (2, 100))
    x = nn.relu(x)
    loc_raw = nn.Dense(2)(x[..., 0, :])
    scale_raw = 0.5 * nn.Dense(2)(x[..., 1, :])
    return nn.tanh(loc_raw), jnp.exp(scale_raw)


class DecoderWhat(nn.Module):

  @nn.compact
  def __call__(self, z_what):
    x = nn.Dense(200)(z_what)
    x = nn.relu(x)
    x = nn.Dense(400)(x)
    x = nn.relu(x)
    x = nn.Dense(784)(x)
    logits = x.reshape(x.shape[:-1] + (28, 28))
    return nn.sigmoid(logits)


class BMNISTAutoEncoder(nn.Module):
  digit_mean: jnp.ndarray
  frame_size: int

  def setup(self):
    self.encode_what = EncoderWhat()
    self.encode_where = EncoderWhere()
    self.decode_what = DecoderWhat()

  def __call__(self, frames):
    # Heuristic procedure to setup initial parameters.
    frames_conv = conv2d(frames, self.digit_mean)
    z_where, _ = self.encode_where(frames_conv)

    digits = crop_frames(frames, z_where, 28)
    z_what, _ = self.encode_what(digits)

    digit_recon = self.decode_what(z_what)
    frames_recon = embed_digits(digit_recon, z_where, self.frame_size)
    return frames_recon


# %%
# Then, we define the target and kernels as in Section 6.4.


def bmnist_target(network, inputs, D=2, T=10):
  z_what = numpyro.sample(
      "z_what", dist.Normal(0, 1).expand([D, 10]).to_event()
  )
  digits = network.decode_what(z_what)  # can cache this

  z_where = []
  # p = []
  for d in range(D):
    z_where_d = []
    z_where_d_t = jnp.zeros(2)
    for t in range(T):
      scale = 1 if t == 0 else 0.1
      z_where_d_t = numpyro.sample(
          f"z_where_{d}_{t}", dist.Normal(z_where_d_t, scale).to_event(1)
      )
      z_where_d.append(z_where_d_t)
    z_where_d = jnp.stack(z_where_d, -2)
    z_where.append(z_where_d)
  z_where = jnp.stack(z_where, -3)

  p = embed_digits(digits, z_where, network.frame_size)
  p = dist.util.clamp_probs(p.sum(-4))  # sum across digits
  frames = numpyro.sample("frames", dist.Bernoulli(p).to_event(3), obs=inputs)

  out = {
      "frames": frames,
      "frames_recon": p,
      "z_what": z_what,
      "digits": jax.lax.stop_gradient(digits),
      **{f"z_where_{t}": z_where[..., t, :] for t in range(T)},
  }
  return (out,)


def kernel_where(network, inputs, D=2, t=0):
  if not isinstance(inputs, dict):
    inputs = {
        "frames": inputs,
        "digits": jnp.repeat(jnp.expand_dims(network.digit_mean, -3), D, -3),
    }

  frame = inputs["frames"][..., t, :, :]
  z_where_t = []
  for d in range(D):
    digit = inputs["digits"][..., d, :, :]
    x_conv = conv2d(frame, digit)
    loc, scale = network.encode_where(x_conv)
    z_where_d_t = numpyro.sample(
        f"z_where_{d}_{t}", dist.Normal(loc, scale).to_event(1)
    )
    z_where_t.append(z_where_d_t)
    frame_recon = embed_digits(digit, z_where_d_t, network.frame_size)
    frame = frame - frame_recon
  z_where_t = jnp.stack(z_where_t, -2)

  out = {**inputs, **{f"z_where_{t}": z_where_t}}
  return (out,)


def kernel_what(network, inputs, T=10):
  z_where = jnp.stack([inputs[f"z_where_{t}"] for t in range(T)], -2)
  digits = crop_frames(inputs["frames"], z_where, 28)
  loc, scale = network.encode_what(digits)
  z_what = numpyro.sample("z_what", dist.Normal(loc, scale).to_event(2))

  out = {**inputs, **{"z_what": z_what}}
  return (out,)


# %%
# Finally, we create the dmm inference program, define the loss function,
# run the training loop, and plot the results.


def make_bmnist(params, bmnist_net, T=10, num_sweeps=5, num_particles=10):
  network = coix.util.BindModule(bmnist_net, params)
  # Add particle dimension and construct a program.
  make_particle_plate = lambda: numpyro.plate("particle", num_particles, dim=-2)
  target = make_particle_plate()(partial(bmnist_target, network, D=2, T=T))
  kernels = []
  for t in range(T):
    kernels.append(
        make_particle_plate()(partial(kernel_where, network, D=2, t=t))
    )
  kernels.append(make_particle_plate()(partial(kernel_what, network, T=T)))
  program = coix.algo.apgs(target, kernels, num_sweeps=num_sweeps)
  return program


def loss_fn(params, key, batch, bmnist_net, num_sweeps, num_particles):
  # Prepare data for the program.
  shuffle_rng, rng_key = random.split(key)
  batch = random.permutation(shuffle_rng, batch, axis=1)
  T = batch.shape[-3]

  # Run the program and get metrics.
  program = make_bmnist(params, bmnist_net, T, num_sweeps, num_particles)
  _, _, metrics = coix.traced_evaluate(program, seed=rng_key)(batch)
  for metric_name in ["log_Z", "log_density", "loss"]:
    metrics[metric_name] = metrics[metric_name] / batch.shape[0]
  return metrics["loss"], metrics


def main(args):
  lr = args.learning_rate
  num_steps = args.num_steps
  batch_size = args.batch_size
  num_sweeps = args.num_sweeps
  num_particles = args.num_particles

  train_ds = load_dataset(is_training=True, batch_size=batch_size)
  test_ds = load_dataset(is_training=False, batch_size=1)
  digit_mean = get_digit_mean()

  test_data = next(test_ds)
  frame_size = test_data.shape[-1]
  bmnist_net = BMNISTAutoEncoder(digit_mean=digit_mean, frame_size=frame_size)
  init_params = bmnist_net.init(jax.random.PRNGKey(0), test_data[0])
  bmnist_params, _ = coix.util.train(
      partial(
          loss_fn,
          bmnist_net=bmnist_net,
          num_sweeps=num_sweeps,
          num_particles=num_particles,
      ),
      init_params,
      optax.adam(lr),
      num_steps,
      train_ds,
  )

  T_test = test_data.shape[-3]
  program = make_bmnist(
      bmnist_params, bmnist_net, T_test, num_sweeps, num_particles
  )
  out, _, _ = coix.traced_evaluate(program, seed=jax.random.PRNGKey(1))(
      test_data
  )
  out = out[0]

  prop_cycle = plt.rcParams["axes.prop_cycle"]
  colors = prop_cycle.by_key()["color"]
  fig, axes = plt.subplots(1, 2, figsize=(12, 6))

  def animate(i):
    axes[0].cla()
    axes[0].imshow(test_data[0, i])
    axes[1].cla()
    axes[1].imshow(out["frames_recon"][0, 0, i])
    for d in range(2):
      where = 0.5 * (out[f"z_where_{i}"][0, 0, d] + 1) * (frame_size - 28) - 0.5
      color = colors[d]
      axes[0].add_patch(
          Rectangle(where, 28, 28, edgecolor=color, lw=3, fill=False)
      )

  plt.rc("animation", html="jshtml")
  plt.tight_layout()
  ani = animation.FuncAnimation(fig, animate, frames=range(20), interval=300)
  writer = animation.PillowWriter(fps=15)
  ani.save("bmnist.gif", writer=writer)
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Annealing example")
  parser.add_argument("--batch-size", nargs="?", default=5, type=int)
  parser.add_argument("--num-sweeps", nargs="?", default=5, type=int)
  parser.add_argument("--num_particles", nargs="?", default=10, type=int)
  parser.add_argument("--learning-rate", nargs="?", default=1e-4, type=float)
  parser.add_argument("--num-steps", nargs="?", default=20000, type=int)
  parser.add_argument(
      "--device", default="gpu", type=str, help='use "cpu" or "gpu".'
  )
  args = parser.parse_args()

  tf.config.experimental.set_visible_devices([], "GPU")  # Disable GPU for TF.
  numpyro.set_platform(args.device)

  main(args)
