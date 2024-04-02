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
Example: Gaussian Mixture Model in NumPyro
==========================================

This example illustrates how to construct an inference program based on the APGS
sampler [1] for GMM. The details of GMM can be found in the sections 6.2 and
F.1 of the reference. We will use the NumPyro (default) backend for this
example.

**References**

    1. Wu, Hao, et al. Amortized population Gibbs samplers with neural
       sufficient statistics. ICML 2020.

"""

import argparse
from functools import partial

import coix
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.ops.indexing import Vindex
import optax
import tensorflow as tf

# %%
# First, let's simulate a synthetic dataset of 2D Gaussian mixtures.


def simulate_clusters(num_instances=1, N=60, seed=0):
  np.random.seed(seed)
  tau = np.random.gamma(2, 0.5, (num_instances, 4, 2))
  mu_base = np.random.normal(0, 1, (num_instances, 4, 2))
  mu = mu_base / np.sqrt(0.1 * tau)
  c = np.random.choice(np.arange(3), (num_instances, N))
  mu_ = np.take_along_axis(mu, c[..., None], axis=1)
  tau_ = np.take_along_axis(tau, c[..., None], axis=1)
  eps = np.random.normal(0, 1, (num_instances, N, 2))
  x = mu_ + eps / np.sqrt(tau_)
  return x, c


def load_dataset(split, *, batch_size):
  if split == "train":
    num_data = 20000
    num_points = 60
    seed = 0
  else:
    num_data = batch_size
    num_points = 100
    seed = 1
  data, label = simulate_clusters(num_data, num_points, seed=seed)
  if split == "train":
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.repeat()
    ds = ds.shuffle(10 * batch_size, seed=0)
  else:
    ds = tf.data.Dataset.from_tensor_slices((data, label))
    ds = ds.repeat()
  ds = ds.batch(batch_size)
  return ds.as_numpy_iterator()


# %%
# Next, we define the neural proposals for the Gibbs kernels.


class GMMEncoderMeanTau(nn.Module):

  @nn.compact
  def __call__(self, x):
    s = nn.Dense(2)(x)

    t = nn.Dense(3)(x)
    t = nn.softmax(t, -1)

    s, t = jnp.expand_dims(s, -2), jnp.expand_dims(t, -1)
    N = t.sum(-3)
    x = (t * s).sum(-3)
    x2 = (t * s**2).sum(-3)
    mu0, nu0, alpha0, beta0 = (0, 0.1, 2, 2)
    alpha = alpha0 + 0.5 * N
    beta = (
        beta0
        + 0.5 * (x2 - x**2 / N)
        + 0.5 * N * nu0 / (N + nu0) * (x / N - mu0) ** 2
    )
    mu = (mu0 * nu0 + x) / (nu0 + N)
    nu = nu0 + N
    return alpha, beta, mu, nu


class GMMEncoderC(nn.Module):

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(32)(x)
    x = nn.tanh(x)
    logits = nn.Dense(1)(x).squeeze(-1)
    return logits + jnp.log(jnp.ones(3) / 3)


class GMMEncoder(nn.Module):

  def setup(self):
    self.encode_initial_mean_tau = GMMEncoderMeanTau()
    self.encode_mean_tau = GMMEncoderMeanTau()
    self.encode_c = GMMEncoderC()

  def __call__(self, x):  # N x D
    # Heuristic procedure to setup initial parameters.
    alpha, beta, mean, _ = self.encode_initial_mean_tau(x)  # M x D
    tau = alpha / beta  # M x D

    concatenate_fn = lambda x, m, t: jnp.concatenate(
        [x, m, t], axis=-1
    )  # N x M x 3D
    xmt = jax.vmap(
        jax.vmap(concatenate_fn, in_axes=(None, 0, 0)), in_axes=(0, None, None)
    )(x, mean, tau)
    logits = self.encode_c(xmt)  # N x D
    c = jnp.argmax(logits, -1)  # N

    xc = jnp.concatenate([x, jax.nn.one_hot(c, 3)], axis=-1)
    return self.encode_mean_tau(xc)


# %%
# Then, we define the target and kernels as in Section 6.2.


def gmm_target(network, inputs):
  with numpyro.plate("N", inputs.shape[-2], dim=-1):
    tau = numpyro.sample("tau", dist.Gamma(2, 2).expand([3, 2]).to_event())
    mean = numpyro.sample("mean", dist.Normal(0, 1 / jnp.sqrt(tau * 0.1)))
    c = numpyro.sample("c", dist.Categorical(probs=jnp.ones(4) / 4))
    loc = Vindex(mean)[..., c, :]
    scale = 1 / jnp.sqrt(Vindex(tau)[..., c, :])
    x = numpyro.sample("x", dist.Normal(loc, scale).to_event(1), obs=inputs)

  out = {"mean": mean, "tau": tau, "c": c, "x": x}
  return out,


def dmm_target(network, inputs):
  mu = numpyro.sample("mu", dist.Normal(0, 10).expand([4, 2]).to_event())
  with numpyro.plate("N", inputs.shape[-2], dim=-1):
    c = numpyro.sample("c", dist.Categorical(probs=jnp.ones(4) / 4))
    h = numpyro.sample("h", dist.Beta(1, 1))
    x_recon = network.decode_h(h) + Vindex(mu)[..., c, :]
    x = numpyro.sample("x", dist.Normal(x_recon, 0.1).to_event(1), obs=inputs)

  out = {"mu": mu, "c": c, "h": h, "x_recon": x_recon, "x": x}
  return (out,)


def gmm_kernel_mean_tau(network, key, inputs):
  if not isinstance(inputs, dict):
    inputs = {"x": inputs}
  key_out, key_mean, key_tau = random.split(key, 3)

  if "c" in inputs:
    c = jax.nn.one_hot(inputs["c"], 3)
    xc = jnp.concatenate([inputs["x"], c], -1)
    alpha, beta, mu, nu = network.encode_mean_tau(xc)
  else:
    alpha, beta, mu, nu = network.encode_initial_mean_tau(inputs["x"])
  tau = coix.rv(dist.Gamma(alpha, beta), name="tau")(key_tau)
  mean = coix.rv(dist.Normal(mu, 1 / jnp.sqrt(tau * nu)), name="mean")(key_mean)

  out = {**inputs, **{"mean": mean, "tau": tau}}
  return key_out, out


def gmm_kernel_c(network, key, inputs):
  key_out, key_c = random.split(key, 2)

  concatenate_fn = lambda x, m, t: jnp.concatenate([x, m, t], axis=-1)
  xmt = jax.vmap(
      jax.vmap(concatenate_fn, in_axes=(None, 0, 0)), in_axes=(0, None, None)
  )(inputs["x"], inputs["mean"], inputs["tau"])
  logits = network.encode_c(xmt)
  c = coix.rv(dist.Categorical(logits=logits), name="c")(key_c)

  out = {**inputs, **{"c": c}}
  return key_out, out


# %%
# Finally, we create the gmm inference program, define the loss function,
# run the training loop, and plot the results.


def make_gmm(params, num_sweeps):
  network = coix.util.BindModule(GMMEncoder(), params)
  # Add particle dimension and construct a program.
  target = jax.vmap(partial(gmm_target, network))
  kernels = [
      jax.vmap(partial(gmm_kernel_mean_tau, network)),
      jax.vmap(partial(gmm_kernel_c, network)),
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
  program = make_gmm(params, num_sweeps)
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

  train_ds = load_dataset("train", batch_size=batch_size)
  test_ds = load_dataset("test", batch_size=batch_size)

  init_params = GMMEncoder().init(jax.random.PRNGKey(0), jnp.zeros((60, 2)))
  gmm_params, _ = coix.util.train(
      partial(loss_fn, num_sweeps=num_sweeps, num_particles=num_particles),
      init_params,
      optax.adam(lr),
      num_steps,
      train_ds,
  )

  program = make_gmm(gmm_params, num_sweeps)
  batch, label = next(test_ds)
  batch = jnp.repeat(batch[:, None], num_particles, axis=1)
  rng_keys = jax.vmap(partial(random.split, num=num_particles))(
      random.split(jax.random.PRNGKey(1), batch.shape[0])
  )
  _, out = jax.vmap(program)(rng_keys, batch)

  fig, axes = plt.subplots(1, 3, figsize=(15, 5))
  for i in range(3):
    n = i
    axes[i].scatter(
        batch[n, 0, :, 0],
        batch[n, 0, :, 1],
        marker=".",
        color=np.array(["c", "m", "y"])[label[n]],
    )
    for j, c in enumerate(["r", "g", "b"]):
      ellipse = Ellipse(
          xy=(out["mean"][n, 0, j, 0], out["mean"][n, 0, j, 1]),
          width=4 / jnp.sqrt(out["tau"][n, 0, j, 0]),
          height=4 / jnp.sqrt(out["tau"][n, 0, j, 1]),
          fc=c,
          alpha=0.3,
      )
      axes[i].add_patch(ellipse)
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Annealing example")
  parser.add_argument("--batch-size", nargs="?", default=20, type=int)
  parser.add_argument("--num-sweeps", nargs="?", default=5, type=int)
  parser.add_argument("--num_particles", nargs="?", default=10, type=int)
  parser.add_argument("--learning-rate", nargs="?", default=2.5e-4, type=float)
  parser.add_argument("--num-steps", nargs="?", default=200000, type=int)
  parser.add_argument(
      "--device", default="gpu", type=str, help='use "cpu" or "gpu".'
  )
  args = parser.parse_args()

  tf.config.experimental.set_visible_devices([], "GPU")  # Disable GPU for TF.
  numpyro.set_platform(args.device)
  coix.set_backend("coix.oryx")

  main(args)
