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
Example: Deep Generative Mixture Model in NumPyro
=================================================

This example illustrates how to construct an inference program based on the APGS
sampler [1] for DMM. The details of DMM can be found in the sections 6.3 and
F.2 of the reference. We will use the NumPyro (default) backend for this
example.

**References**

    1. Wu, Hao, et al. Amortized population Gibbs samplers with neural
       sufficient statistics. ICML 2020.

.. image:: ../_static/dmm.png
    :align: center

"""

import argparse
from functools import partial

import coix
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.ops.indexing import Vindex
import optax
import tensorflow as tf

# %%
# First, let's simulate a synthetic dataset of 2D ring-shaped mixtures.


def simulate_rings(num_instances=1, N=200, seed=0):
  np.random.seed(seed)
  mu = np.random.normal(0, 3, (num_instances, 1, 4, 2))
  angle = np.linspace(0, 2 * np.pi, N // 8, endpoint=False)
  shift = np.random.uniform(
      0, (2 * np.pi) // (N // 8), size=(num_instances, 1, 2, 4)
  )
  angle = angle[:, None, None] + shift
  angle = angle.reshape((num_instances, N // 4, 4))
  loc = np.stack([np.cos(angle), np.sin(angle)], -1)
  noise = np.random.normal(0, 0.1, loc.shape)
  x = (mu + loc + noise).reshape((num_instances, N, 2))
  shuffle_idx = np.random.uniform(size=x.shape[:2] + (1,)).argsort(axis=1)
  return np.take_along_axis(x, shuffle_idx, axis=1)


def load_dataset(split, *, batch_size):
  if split == "train":
    num_data = 20000
    num_points = 200
    seed = 0
  else:
    num_data = batch_size
    num_points = 600
    seed = 1
  data = simulate_rings(num_data, num_points, seed=seed)
  ds = tf.data.Dataset.from_tensor_slices(data)
  ds = ds.repeat()
  if split == "train":
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return ds.as_numpy_iterator()


# %%
# Next, we define the neural proposals for the Gibbs kernels and the neural
# decoder for the generative model.


class EncoderMu(nn.Module):

  @nn.compact
  def __call__(self, x):
    s = nn.Dense(32)(x)
    s = nn.tanh(s)
    s = nn.Dense(8)(s)

    t = nn.Dense(32)(x)
    t = nn.tanh(t)
    t = nn.Dense(4)(t)
    t = nn.softmax(t, -1)

    s, t = jnp.expand_dims(s, -2), jnp.expand_dims(t, -1)
    st = (s * t).sum(-3) / t.sum(-3)

    shape = st.shape[:-1] + (2,)
    x = jnp.concatenate([st, jnp.zeros(shape), jnp.full(shape, 10.0)], -1)
    x = nn.Dense(64)(x)
    x = x.reshape(x.shape[:-1] + (2, 32))
    x = nn.tanh(x)
    loc = nn.Dense(2)(x[..., 0, :])
    scale_raw = 0.5 * nn.Dense(2)(x[..., 1, :])
    return loc, jnp.exp(scale_raw)


class EncoderC(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(32)(x)
    x = nn.relu(x)  # nn.tanh(x)
    logits = nn.Dense(1)(x).squeeze(-1)
    return logits + jnp.log(jnp.ones(4) / 4)


class EncoderH(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(64)(x)
    x = x.reshape(x.shape[:-1] + (2, 32))
    x = nn.tanh(x)
    alpha_raw = nn.Dense(1)(x[..., 0, :]).squeeze(-1)
    beta_raw = nn.Dense(1)(x[..., 1, :]).squeeze(-1)
    return jnp.exp(alpha_raw), jnp.exp(beta_raw)


class DecoderH(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(32)(jnp.expand_dims(x, -1))
    x = nn.tanh(x)
    x = nn.Dense(2)(x)
    angle = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    radius = 1.0  # self.param("radius", nn.initializers.ones, (1,))
    return radius * angle


class DMMAutoEncoder(nn.Module):

  def setup(self):
    self.encode_initial_mu = EncoderMu()
    self.encode_mu = EncoderMu()
    self.encode_c = EncoderC()
    self.encode_h = EncoderH()
    self.decode_h = DecoderH()

  def __call__(self, x):  # N x D
    # Heuristic procedure to setup initial parameters.
    mu, _ = self.encode_initial_mu(x)  # M x D

    xmu = jnp.expand_dims(x, -2) - mu
    logits = self.encode_c(xmu)  # N x M
    c = jnp.argmax(logits, -1)  # N

    loc = mu[c]  # N x D
    alpha, beta = self.encode_h(x - loc)  # N
    h = alpha / (alpha + beta)  # N

    xch = jnp.concatenate([x, jax.nn.one_hot(c, 4), jnp.expand_dims(h, -1)], -1)
    mu, _ = self.encode_mu(xch)  # M x D

    angle = self.decode_h(h)  # N x D
    x_recon = mu[c] + angle  # N x D
    return x_recon


# %%
# Then, we define the target and kernels as in Section 6.3.


def dmm_target(network, inputs):
  mu = numpyro.sample("mu", dist.Normal(0, 10).expand([4, 2]).to_event())
  with numpyro.plate("N", inputs.shape[-2], dim=-1):
    c = numpyro.sample("c", dist.Categorical(probs=jnp.ones(4) / 4))
    h = numpyro.sample("h", dist.Beta(1, 1))
    x_recon = network.decode_h(h) + Vindex(mu)[..., c, :]
    x = numpyro.sample("x", dist.Normal(x_recon, 0.1).to_event(1), obs=inputs)

  out = {"mu": mu, "c": c, "h": h, "x_recon": x_recon, "x": x}
  return (out,)


def dmm_kernel_mu(network, inputs):
  if not isinstance(inputs, dict):
    inputs = {"x": inputs}

  if "c" in inputs:
    x = jnp.broadcast_to(inputs["x"], inputs["h"].shape + (2,))
    c = jax.nn.one_hot(inputs["c"], 4)
    h = jnp.expand_dims(inputs["h"], -1)
    xch = jnp.concatenate([x, c, h], -1)
    loc, scale = network.encode_mu(xch)
  else:
    loc, scale = network.encode_initial_mu(inputs["x"])
  loc, scale = jnp.expand_dims(loc, -3), jnp.expand_dims(scale, -3)
  mu = numpyro.sample("mu", dist.Normal(loc, scale).to_event(2))

  out = {**inputs, **{"mu": mu}}
  return (out,)


def dmm_kernel_c_h(network, inputs):
  x, mu = inputs["x"], inputs["mu"]
  xmu = jnp.expand_dims(x, -2) - mu
  logits = network.encode_c(xmu)
  with numpyro.plate("N", logits.shape[-2], dim=-1):
    c = numpyro.sample("c", dist.Categorical(logits=logits))
    alpha, beta = network.encode_h(inputs["x"] - Vindex(mu)[..., c, :])
    h = numpyro.sample("h", dist.Beta(alpha, beta))

  out = {**inputs, **{"c": c, "h": h}}
  return (out,)


# %%
# Finally, we create the dmm inference program, define the loss function,
# run the training loop, and plot the results.


def make_dmm(params, num_sweeps=5, num_particles=10):
  network = coix.util.BindModule(DMMAutoEncoder(), params)
  # Add particle dimension and construct a program.
  vmap = lambda p: numpyro.plate("particle", num_particles, dim=-3)(p)
  target = vmap(partial(dmm_target, network))
  kernel_mu = vmap(partial(dmm_kernel_mu, network))
  kernel_c_h = vmap(partial(dmm_kernel_c_h, network))
  kernels = [kernel_mu, kernel_c_h]
  program = coix.algo.apgs(target, kernels, num_sweeps=num_sweeps)
  return program


def loss_fn(params, key, batch, num_sweeps, num_particles):
  # Prepare data for the program.
  shuffle_rng, rng_key = random.split(key)
  batch = random.permutation(shuffle_rng, batch, axis=1)

  # Run the program and get metrics.
  program = make_dmm(params, num_sweeps, num_particles)
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

  train_ds = load_dataset("train", batch_size=batch_size)
  test_ds = load_dataset("test", batch_size=batch_size)

  init_params = DMMAutoEncoder().init(
      jax.random.PRNGKey(0), jnp.zeros((200, 2))
  )
  dmm_params, _ = coix.util.train(
      partial(loss_fn, num_sweeps=num_sweeps, num_particles=num_particles),
      init_params,
      optax.adam(lr),
      num_steps,
      train_ds,
  )

  program = make_dmm(dmm_params, num_sweeps, num_particles)
  batch = next(test_ds)
  out, _, _ = coix.traced_evaluate(program, seed=jax.random.PRNGKey(1))(batch)
  out = out[0]

  _, axes = plt.subplots(2, 3, figsize=(15, 10))
  for i in range(3):
    axes[0][i].scatter(out["x"][i, :, 0], out["x"][i, :, 1], marker=".")
    axes[1][i].scatter(
        out["x_recon"][0, i, :, 0],
        out["x_recon"][0, i, :, 1],
        c=out["c"][0, i],
        cmap="Accent",
        marker=".",
    )
    axes[1][i].scatter(
        out["mu"][0, i, 0, :, 0],
        out["mu"][0, i, 0, :, 1],
        c=range(4),
        marker="x",
        cmap="Accent",
    )
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Annealing example")
  parser.add_argument("--batch-size", nargs="?", default=20, type=int)
  parser.add_argument("--num-sweeps", nargs="?", default=5, type=int)
  parser.add_argument("--num_particles", nargs="?", default=10, type=int)
  parser.add_argument("--learning-rate", nargs="?", default=1e-3, type=float)
  parser.add_argument("--num-steps", nargs="?", default=30000, type=int)
  parser.add_argument(
      "--device", default="gpu", type=str, help='use "cpu" or "gpu".'
  )
  args = parser.parse_args()

  tf.config.experimental.set_visible_devices([], "GPU")  # Disable GPU for TF.
  numpyro.set_platform(args.device)

  main(args)
