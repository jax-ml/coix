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

.. image:: ../_static/gmm_oryx.png
    :align: center

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


def broadcast_concatenate(*xs):
  shape = jnp.broadcast_shapes(*[x.shape[:-1] for x in xs])
  xs = [jnp.broadcast_to(x, shape + x.shape[-1:]) for x in xs]
  return jnp.concatenate(xs, -1)


class GMMEncoder(nn.Module):

  def setup(self):
    self.encode_initial_mean_tau = GMMEncoderMeanTau()
    self.encode_mean_tau = GMMEncoderMeanTau()
    self.encode_c = GMMEncoderC()

  def __call__(self, x):  # N x D
    # Heuristic procedure to setup initial parameters.
    alpha, beta, mean, _ = self.encode_initial_mean_tau(x)  # M x D
    tau = alpha / beta  # M x D

    xmt = jax.vmap(broadcast_concatenate, (None, -2, -2), -2)(x, mean, tau)
    logits = self.encode_c(xmt)  # N x D
    c = jnp.argmax(logits, -1)  # N

    xc = jnp.concatenate([x, jax.nn.one_hot(c, 3)], axis=-1)
    return self.encode_mean_tau(xc)


# %%
# Then, we define the target and kernels as in Section 6.2.


def gmm_target(inputs):
  tau = numpyro.sample("tau", dist.Gamma(2, 2).expand([3, 2]).to_event())
  mean = numpyro.sample(
      "mean", dist.Normal(0, 1 / jnp.sqrt(tau * 0.1)).to_event()
  )
  with numpyro.plate("N", inputs.shape[-2], dim=-1):
    c = numpyro.sample("c", dist.Categorical(probs=jnp.ones(4) / 4))
    loc = Vindex(mean)[..., c, :]
    scale = 1 / jnp.sqrt(Vindex(tau)[..., c, :])
    x = numpyro.sample("x", dist.Normal(loc, scale).to_event(1), obs=inputs)

  out = {"mean": mean, "tau": tau, "c": c, "x": x}
  return (out,)


def gmm_kernel_mean_tau(network, inputs):
  if not isinstance(inputs, dict):
    inputs = {"x": inputs}

  if "c" in inputs:
    x = inputs["x"]
    c = jax.nn.one_hot(inputs["c"], 3)
    xc = broadcast_concatenate(x, c)
    alpha, beta, mu, nu = network.encode_mean_tau(xc)
  else:
    alpha, beta, mu, nu = network.encode_initial_mean_tau(inputs["x"])
  alpha, beta, mu, nu = jax.tree_util.tree_map(
      lambda x: jnp.expand_dims(x, -3), (alpha, beta, mu, nu)
  )
  tau = numpyro.sample("tau", dist.Gamma(alpha, beta).to_event(2))
  mean = numpyro.sample(
      "mean", dist.Normal(mu, 1 / jnp.sqrt(tau * nu)).to_event(2)
  )

  out = {**inputs, **{"mean": mean, "tau": tau}}
  return (out,)


def gmm_kernel_c(network, inputs):
  x, mean, tau = inputs["x"], inputs["mean"], inputs["tau"]
  xmt = jax.vmap(broadcast_concatenate, (None, -2, -2), -2)(x, mean, tau)
  logits = network.encode_c(xmt)
  with numpyro.plate("N", logits.shape[-2], dim=-1):
    c = numpyro.sample("c", dist.Categorical(logits=logits))

  out = {**inputs, **{"c": c}}
  return (out,)


# %%
# Finally, we create the gmm inference program, define the loss function,
# run the training loop, and plot the results.


def make_gmm(params, num_sweeps, num_particles):
  network = coix.util.BindModule(GMMEncoder(), params)
  # Add particle dimension and construct a program.
  make_particle_plate = lambda: numpyro.plate("particle", num_particles, dim=-3)
  target = make_particle_plate()(gmm_target)
  kernel_mean_tau = make_particle_plate()(partial(gmm_kernel_mean_tau, network))
  kernel_c = make_particle_plate()(partial(gmm_kernel_c, network))
  kernels = [kernel_mean_tau, kernel_c]
  program = coix.algo.apgs(target, kernels, num_sweeps=num_sweeps)
  return program


def loss_fn(params, key, batch, num_sweeps, num_particles):
  # Prepare data for the program.
  shuffle_rng, rng_key = random.split(key)
  batch = random.permutation(shuffle_rng, batch, axis=1)

  # Run the program and get metrics.
  program = make_gmm(params, num_sweeps, num_particles)
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

  init_params = GMMEncoder().init(jax.random.PRNGKey(0), jnp.zeros((60, 2)))
  gmm_params, _ = coix.util.train(
      partial(loss_fn, num_sweeps=num_sweeps, num_particles=num_particles),
      init_params,
      optax.adam(lr),
      num_steps,
      train_ds,
  )

  program = make_gmm(gmm_params, num_sweeps, num_particles)
  batch, label = next(test_ds)
  out, _, _ = coix.traced_evaluate(program, seed=jax.random.PRNGKey(1))(batch)
  out = out[0]

  _, axes = plt.subplots(2, 3, figsize=(15, 10))
  for i in range(6):
    axes[i // 3][i % 3].scatter(
        batch[i, :, 0],
        batch[i, :, 1],
        marker=".",
        color=np.array(["c", "m", "y"])[label[i]],
    )
    for j, c in enumerate(["r", "g", "b"]):
      ellipse = Ellipse(
          xy=(out["mean"][0, i, 0, j, 0], out["mean"][0, i, 0, j, 1]),
          width=4 / jnp.sqrt(out["tau"][0, i, 0, j, 0]),
          height=4 / jnp.sqrt(out["tau"][0, i, 0, j, 1]),
          fc=c,
          alpha=0.3,
      )
      axes[i // 3][i % 3].add_patch(ellipse)
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

  main(args)
