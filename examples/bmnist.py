# Copyright 2023 The coix Authors.
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

# TODO: Refactor using numpyro backend. The current code is likely not working yet.

import argparse
from functools import partial
import sys

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import flax
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
import optax

import coix
import numpyro
import numpyro.distributions as dist


batch_size = 5
T = 10  # using 10 time steps for training and 20 time steps for testing


def load_dataset(*, is_training, batch_size):
    ds, ds_info = tfds.load("moving_mnist:1.0.0", split="test", with_info=True)
    ds = ds.cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
        map_fn = lambda x: x["image_sequence"][..., :T, :, :, 0] / 255
    else:
        map_fn = lambda x: x["image_sequence"][..., 0] / 255
    ds = ds.batch(batch_size)
    ds = ds.map(map_fn)
    return iter(tfds.as_numpy(ds)), ds_info


def get_digit_mean():
    ds, ds_info = tfds.load("mnist:3.0.1", split="train", with_info=True)
    ds = tfds.as_numpy(ds.batch(ds_info.splits["test"].num_examples))
    digit_mean = next(iter(ds))["image"].squeeze(-1).mean(axis=0)
    return digit_mean / 255


train_ds, ds_info = load_dataset(is_training=True, batch_size=batch_size)
test_ds, _ = load_dataset(is_training=False, batch_size=1)
digit_mean = get_digit_mean()
frame_size = ds_info.features['image_sequence'].shape[-2]
frame_length = ds_info.features['image_sequence']._length
print("Frame length: ", frame_length)
print("Frame size:", frame_size)
print("Digit shape:", digit_mean.shape)


### Autoencoder

def scale_and_translate(image, where, out_size):
  translate = abs(image.shape[-1] - out_size) * (where[..., ::-1] + 1) / 2
  return jax.image.scale_and_translate(
      image, (out_size, out_size), (0, 1),
      jnp.ones(2),
      translate,
      method="cubic",
      antialias=False)


def crop_frames(frames, z_where, digit_size=28):
  # frames:           time.frame_size.frame_size
  # z_where: (digits).time.2
  # out:     (digits).time.digit_size.digit_size
  crop_fn = partial(scale_and_translate, out_size=digit_size)
  if z_where.ndim == 2:
    return jax.vmap(crop_fn)(frames, z_where)
  return jax.vmap(jax.vmap(crop_fn), in_axes=(None, 0))(frames, z_where)


def embed_digits(digits, z_where, frame_size=64):
  # digits:  (digits).      .digit_size.digit_size
  # z_where: (digits).(time).2
  # out:     (digits).(time).frame_size.frame_size
  embed_fn = partial(scale_and_translate, out_size=frame_size)
  if digits.ndim == 2:
    if z_where.ndim == 1:
      return embed_fn(digits, z_where)
    return jax.vmap(embed_fn, in_axes=(None, 0))(digits, z_where)
  return jax.vmap(jax.vmap(embed_fn, in_axes=(None, 0)))(digits, z_where)


def conv2d(frames, digits):
  # frames:          (time).frame_size.frame_size
  # digits: (digits).      .digit_size.digit_size
  # out:    (digits).(time).conv_size .conv_size
  conv2d_fn = partial(jax.scipy.signal.convolve2d, mode="valid")
  if frames.ndim == 2:
    if digits.ndim == 2:
      return conv2d_fn(frames, digits)
    return jax.vmap(conv2d_fn, in_axes=(None, 0))(frames, digits)
  return jax.vmap(conv2d_fn, in_axes=(0, None))(frames, digits)


class EncoderWhat(nn.Module):

  @nn.compact
  def __call__(self, digits):
    x = digits.reshape(digits.shape[:-2] + (-1,))
    x = nn.Dense(400)(x)
    x = nn.relu(x)
    x = nn.Dense(200)(x)
    x = nn.relu(x)

    x = x.sum(-2)  # sum across time
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


### Model and kernels

test_key = random.PRNGKey(0)
test_data = jnp.zeros((frame_length,) + (frame_size, frame_size))
bmnist_net = BMNISTAutoEncoder(digit_mean=digit_mean, frame_size=frame_size)
init_params = bmnist_net.init(test_key, test_data)
test_network = coix.util.BindModule(bmnist_net, init_params)


def bmnist_target(network, key, inputs, D=2, T=10):
  key_out, key_what, key_where = random.split(key, 3)

  z_what = coix.rv(dist.Normal(0, 1).expand([D, 10]), name="z_what")(key_what)
  digits = network.decode_what(z_what)  # can cache this

  z_where = []
  for d in range(D):
    z_where_d = []
    z_where_d_t = jnp.zeros(2)
    for t in range(T):
      scale = 1 if t == 0 else 0.1
      key_d_t = random.fold_in(key_where, d * T + t)
      name = f"z_where_{d}_{t}"
      z_where_d_t = coix.rv(dist.Normal(z_where_d_t, scale), name=name)(key_d_t)
      z_where_d.append(z_where_d_t)
    z_where.append(jnp.stack(z_where_d, -2))
  z_where = jnp.stack(z_where, -3)

  p = embed_digits(digits, z_where, network.frame_size)
  p = dist.util.clamp_probs(p.sum(-4))  # sum across digits
  frames = coix.rv(dist.Bernoulli(p), obs=inputs, name="frames")

  out = {
      "frames": frames,
      "frames_recon": p,
      "z_what": z_what,
      "digits": jax.lax.stop_gradient(digits),
      **{f"z_where_{t}": z_where[:, t, :] for t in range(T)}
  }
  return key_out, out


_, p_out = bmnist_target(test_network, test_key, test_data, T=frame_length)


def kernel_where(network, key, inputs, D=2, t=0):
  if not isinstance(inputs, dict):
    inputs = {
        "frames": inputs,
        "digits": jnp.repeat(jnp.expand_dims(network.digit_mean, -3), D, -3)
    }
  key_out, key_where = random.split(key)

  frame = inputs["frames"][t, :, :]
  z_where_t = []
  key_where = random.split(key_where, D)
  for d, key_where_d in enumerate(key_where):
    digit = inputs["digits"][d, :, :]
    x_conv = conv2d(frame, digit)
    loc, scale = network.encode_where(x_conv)
    name = f"z_where_{d}_{t}"
    z_where_d_t = coix.rv(dist.Normal(loc, scale), name=name)(key_where_d)
    z_where_t.append(z_where_d_t)
    frame_recon = embed_digits(digit, z_where_d_t, network.frame_size)
    frame = frame - frame_recon
  z_where_t = jnp.stack(z_where_t, -2)

  out = {**inputs, **{f"z_where_{t}": z_where_t}}
  return key_out, out


_, k1_initial_out = kernel_where(test_network, test_key, test_data)
_, k1_out = kernel_where(test_network, test_key, p_out)


def kernel_what(network, key, inputs, T=10):
  key_out, key_what = random.split(key)

  z_where = jnp.stack([inputs[f"z_where_{t}"] for t in range(T)], -2)
  digits = crop_frames(inputs["frames"], z_where, 28)
  loc, scale = network.encode_what(digits)
  z_what = coix.rv(dist.Normal(loc, scale), name="z_what")(key_what)

  out = {**inputs, **{"z_what": z_what}}
  return key_out, out

_, k2_out = kernel_what(test_network, test_key, p_out, T=frame_length)


num_sweeps = 5
num_particles = 10


def make_bmnist(params, T=10):
  network = coix.util.BindModule(bmnist_net, params)
  # Add particle dimension and construct a program.
  target = jax.vmap(partial(bmnist_target, network, D=2, T=T))
  kernels = []
  for t in range(T):
    kernels.append(jax.vmap(partial(kernel_where, network, D=2, t=t)))
  kernels.append(jax.vmap(partial(kernel_what, network, T=T)))
  program = coix.algo.apgs(target, kernels, num_sweeps=num_sweeps)
  return program
a

def loss_fn(params, key, batch):
  # Prepare data for the program.
  shuffle_rng, rng_key = random.split(key)
  batch = random.permutation(shuffle_rng, batch, axis=1)
  batch_rng = random.split(rng_key, batch.shape[0])
  batch = jnp.repeat(batch[:, None], num_particles, axis=1)
  rng_keys = jax.vmap(partial(random.split, num=num_particles))(batch_rng)

  # Run the program and get metrics.
  program = make_bmnist(params)
  _, _, metrics = jax.vmap(coix.traced_evaluate(program))(rng_keys, batch)
  metrics = jax.tree_util.tree_map(jnp.mean, metrics)  # mean across batch
  return metrics["loss"], metrics


lr = 1e-4
num_steps = 200000
bmnist_params, _ = coix.util.train(
    loss_fn, init_params, optax.adam(lr), num_steps, train_ds)


program = make_bmnist(bmnist_params, T=frame_length)
batch = jnp.repeat(next(test_ds)[:1], num_particles, axis=0)
rng_keys = random.split(random.PRNGKey(1), num_particles)
_, out = program(rng_keys, batch)
batch.shape, out["frames_recon"].shape


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
fig, axes = plt.subplots(1, 2, figsize=(12, 6))


def animate(i):
  n = 2
  axes[0].cla()
  axes[0].imshow(batch[n][i])
  axes[1].cla()
  axes[1].imshow(out["frames_recon"][n, i])
  for d in range(2):
    where = 0.5 * (out[f"z_where_{i}"][n, d] + 1) * (frame_size - 28) - 0.5
    color = colors[d]
    axes[0].add_patch(
        Rectangle(where, 28, 28, edgecolor=color, lw=3, fill=False))


plt.rc("animation", html="jshtml")
anim = FuncAnimation(fig, animate, frames=range(20), interval=300)
plt.close()
anim
