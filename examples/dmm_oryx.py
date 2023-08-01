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
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

# Data


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


def load_dataset(split, *, is_training, batch_size):
  num_data = 20000 if is_training else batch_size
  num_points = 200 if is_training else 600
  seed = 0 if is_training else 1
  data = simulate_rings(num_data, num_points, seed=seed)
  ds = tf.data.Dataset.from_tensor_slices(data)
  ds = ds.cache().repeat()
  if is_training:
    ds = ds.shuffle(10 * batch_size, seed=0)
  ds = ds.batch(batch_size)
  return iter(tfds.as_numpy(ds))


### Autoencoder


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
    x = nn.tanh(x)
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
    return angle


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

    concatenate_fn = lambda x, m: jnp.concatenate([x, m], axis=-1)
    xmu = jax.vmap(jax.vmap(concatenate_fn, (None, 0)), (0, None))(x, mu)
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


### Model and kernels


def dmm_target(network, key, inputs):
  key_out, key_mu, key_c, key_h = random.split(key, 4)
  N = inputs.shape[-2]

  mu = coix.rv(dist.Normal(0, 10).expand([4, 2]), name="mu")(key_mu)
  c = coix.rv(dist.DiscreteUniform(0, 3).expand([N]), name="c")(key_c)
  h = coix.rv(dist.Beta(1, 1).expand([N]), name="h")(key_h)
  x_recon = mu[c] + network.decode_h(h)
  x = coix.rv(dist.Normal(x_recon, 0.1), obs=inputs, name="x")

  out = {"mu": mu, "c": c, "h": h, "x_recon": x_recon, "x": x}
  return key_out, out


def dmm_kernel_mu(network, key, inputs):
  if not isinstance(inputs, dict):
    inputs = {"x": inputs}
  key_out, key_mu = random.split(key)

  if "c" in inputs:
    c = jax.nn.one_hot(inputs["c"], 4)
    h = jnp.expand_dims(inputs["h"], -1)
    xch = jnp.concatenate([inputs["x"], c, h], -1)
    loc, scale = network.encode_mu(xch)
  else:
    loc, scale = network.encode_initial_mu(inputs["x"])
  mu = coix.rv(dist.Normal(loc, scale), name="mu")(key_mu)

  out = {**inputs, **{"mu": mu}}
  return key_out, out


def dmm_kernel_c_h(network, key, inputs):
  key_out, key_c, key_h = random.split(key, 3)

  concatenate_fn = lambda x, m: jnp.concatenate([x, m], axis=-1)
  xmu = jax.vmap(jax.vmap(concatenate_fn, (None, 0)), (0, None))(
      inputs["x"], inputs["mu"]
  )
  logits = network.encode_c(xmu)
  c = coix.rv(dist.Categorical(logits=logits), name="c")(key_c)
  alpha, beta = network.encode_h(inputs["x"] - inputs["mu"][c])
  h = coix.rv(dist.Beta(alpha, beta), name="h")(key_h)

  out = {**inputs, **{"c": c, "h": h}}
  return key_out, out


### Train


def make_dmm(params, num_sweeps):
  network = coix.util.BindModule(DMMAutoEncoder(), params)
  # Add particle dimension and construct a program.
  target = jax.vmap(partial(dmm_target, network))
  kernels = [
      jax.vmap(partial(dmm_kernel_mu, network)),
      jax.vmap(partial(dmm_kernel_c_h, network)),
  ]
  program = coix.algo.apgs(target, kernels, num_sweeps=num_sweeps)
  return program


def loss_fn(params, key, batch, num_sweeps, num_particles):
  # Prepare data for the program.
  shuffle_rng, rng_key = random.split(key)
  batch = random.permutation(shuffle_rng, batch, axis=1)
  batch_rng = random.split(rng_key, batch.shape[0])
  batch = jnp.repeat(batch[:, None], num_particles, axis=1)
  rng_keys = jax.vmap(partial(random.split, num=num_particles))(batch_rng)

  # Run the program and get metrics.
  program = make_dmm(params, num_sweeps)
  _, _, metrics = jax.vmap(coix.traced_evaluate(program))(rng_keys, batch)
  metrics = jax.tree_util.tree_map(
      partial(jnp.mean, axis=0), metrics
  )  # mean across batch
  return metrics["loss"], metrics


def main(args):
  lr = args.learning_rate
  num_steps = args.num_steps
  batch_size = args.batch_size
  num_sweeps = args.num_sweeps
  num_particles = args.num_particles

  train_ds = load_dataset("train", is_training=True, batch_size=batch_size)
  test_ds = load_dataset("test", is_training=False, batch_size=batch_size)

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

  program = make_dmm(dmm_params, num_sweeps)
  batch = jnp.repeat(next(test_ds)[:, None], num_particles, axis=1)
  rng_keys = jax.vmap(partial(random.split, num=num_particles))(
      random.split(jax.random.PRNGKey(1), batch.shape[0])
  )
  _, out = jax.vmap(program)(rng_keys, batch)
  batch.shape, out["x_recon"].shape

  fig, axes = plt.subplots(2, 3, figsize=(15, 10))
  for i in range(3):
    n = i
    axes[0][i].scatter(out["x"][n, 0, :, 0], out["x"][n, 0, :, 1], marker=".")
    axes[1][i].scatter(
        out["x_recon"][n, 0, :, 0],
        out["x_recon"][n, 0, :, 1],
        c=out["c"][n, 0],
        cmap="Accent",
        marker=".",
    )
    axes[1][i].scatter(
        out["mu"][n, 0, :, 0],
        out["mu"][n, 0, :, 1],
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
  parser.add_argument("--learning-rate", nargs="?", default=1e-4, type=float)
  parser.add_argument("--num-steps", nargs="?", default=300000, type=int)
  parser.add_argument(
      "--device", default="gpu", type=str, help='use "cpu" or "gpu".'
  )
  args = parser.parse_args()

  tf.config.experimental.set_visible_devices([], "GPU")  # Disable GPU for TF.
  numpyro.set_platform(args.device)
  coix.set_backend("coix.oryx")

  main(args)
