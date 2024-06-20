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
Example: Annealed Variational Inference in Oryx
===============================================

This example illustrates how to construct an inference program based on the NVI
algorithm [1] for AVI. The details of AVI can be found in the sections E.1 of
the reference. We will use the Oryx backend for this example.

**References**

    1. Zimmermann, Heiko, et al. "Nested variational inference." NeuRIPS 2021.

.. image:: ../_static/anneal_oryx.png
    :align: center

"""

import argparse
from functools import partial

import coix
import coix.oryx as coryx
import flax
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax

# %%
# First, we define the neural networks for the targets and kernels.


class AnnealKernel(nn.Module):

  @nn.compact
  def __call__(self, x):
    h = nn.Dense(50)(x)
    h = nn.relu(h)
    loc = nn.Dense(2, kernel_init=nn.initializers.zeros)(h) + x
    scale_raw = nn.Dense(2, kernel_init=nn.initializers.zeros)(h)
    return loc, nn.softplus(scale_raw)


class AnnealDensity(nn.Module):
  M = 8

  @nn.compact
  def __call__(self, x, index=0):
    beta_raw = self.param("beta_raw", lambda _: -jnp.ones(self.M - 2))
    beta = nn.sigmoid(
        beta_raw[0] + jnp.pad(jnp.cumsum(nn.softplus(beta_raw[1:])), (1, 0))
    )
    beta = jnp.pad(beta, (1, 1), constant_values=(0, 1))
    beta_k = beta[index]

    angles = 2 * jnp.arange(1, self.M + 1) * jnp.pi / self.M
    mu = 10 * jnp.stack([jnp.sin(angles), jnp.cos(angles)], -1)
    sigma = jnp.sqrt(0.5)
    target_density = nn.logsumexp(
        dist.Normal(mu, sigma).log_prob(x[..., None, :]).sum(-1), -1
    )
    init_proposal = dist.Normal(0, 5).log_prob(x).sum(-1)
    return beta_k * target_density + (1 - beta_k) * init_proposal


class AnnealKernelList(nn.Module):
  M = 8

  @nn.compact
  def __call__(self, x, index=0):
    if self.is_mutable_collection("params"):
      vmap_net = nn.vmap(
          AnnealKernel, variable_axes={"params": 0}, split_rngs={"params": True}
      )
      out = vmap_net(name="kernel")(
          jnp.broadcast_to(x, (self.M - 1,) + x.shape)
      )
      return jax.tree.map(lambda x: x[index], out)
    params = self.scope.get_variable("params", "kernel")
    params_i = jax.tree.map(lambda x: x[index], params)
    return AnnealKernel(name="kernel").apply(
        flax.core.freeze({"params": params_i}), x
    )


class AnnealNetwork(nn.Module):

  def setup(self):
    self.forward_kernels = AnnealKernelList()
    self.reverse_kernels = AnnealKernelList()
    self.anneal_density = AnnealDensity()

  def __call__(self, x):
    self.reverse_kernels(x)
    self.anneal_density(x)
    return self.forward_kernels(x)


# %%
# Then, we define the targets and kernels as in Section E.1.


def anneal_target(network, key, k=0):
  key_out, key = random.split(key)
  x = coryx.rv(dist.Normal(0, 5).expand([2]).mask(False), name="x")(key)
  coryx.factor(network.anneal_density(x, index=k), name="anneal_density")
  return key_out, {"x": x}


def anneal_forward(network, key, inputs, k=0):
  mu, sigma = network.forward_kernels(inputs["x"], index=k)
  return coryx.rv(dist.Normal(mu, sigma), name="x")(key)


def anneal_reverse(network, key, inputs, k=0):
  mu, sigma = network.reverse_kernels(inputs["x"], index=k)
  return coryx.rv(dist.Normal(mu, sigma), name="x")(key)


# %%
# Finally, we create the anneal inference program, define the loss function,
# run the training loop, and plot the results.


def make_anneal(params, unroll=False):
  network = coix.util.BindModule(AnnealNetwork(), params)
  # Add particle dimension and construct a program.
  targets = lambda k: jax.vmap(partial(anneal_target, network, k=k))
  forwards = lambda k: jax.vmap(partial(anneal_forward, network, k=k))
  reverses = lambda k: jax.vmap(partial(anneal_reverse, network, k=k))
  if unroll:  # to unroll the algorithm, we provide a list of programs
    targets = [targets(k) for k in range(8)]
    forwards = [forwards(k) for k in range(7)]
    reverses = [reverses(k) for k in range(7)]
  program = coix.algo.nvi_rkl(targets, forwards, reverses, num_targets=8)
  return program


def loss_fn(params, key, num_particles, unroll=False):
  # Prepare data for the program.
  rng_keys = random.split(key, num_particles)

  # Run the program and get metrics.
  program = make_anneal(params, unroll=unroll)
  _, _, metrics = coix.traced_evaluate(program)(rng_keys)
  return metrics["loss"], metrics


def main(args):
  lr = args.learning_rate
  num_steps = args.num_steps
  num_particles = args.num_particles
  unroll = args.unroll_loop

  anneal_net = AnnealNetwork()
  init_params = anneal_net.init(random.PRNGKey(0), jnp.zeros(2))

  anneal_params, _ = coix.util.train(
      partial(loss_fn, num_particles=num_particles, unroll=unroll),
      init_params,
      optax.adam(lr),
      num_steps,
      jit_compile=True,
  )

  rng_keys = random.split(random.PRNGKey(1), 100000).reshape((100, 1000, 2))
  _, trace, metrics = coix.traced_evaluate(
      jax.vmap(make_anneal(anneal_params, unroll=unroll))
  )(rng_keys)

  metrics.pop("log_weight")
  anneal_metrics = jax.tree.map(lambda x: round(float(jnp.mean(x)), 4), metrics)
  print(anneal_metrics)

  plt.figure(figsize=(8, 8))
  x = trace["x"]["value"].reshape((-1, 2))
  H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=100)
  plt.imshow(H.T)
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Annealing example")
  parser.add_argument("--num_particles", nargs="?", default=36, type=int)
  parser.add_argument("--learning-rate", nargs="?", default=1e-3, type=float)
  parser.add_argument("--num-steps", nargs="?", default=20000, type=int)
  parser.add_argument("--unroll-loop", action="store_true")
  parser.add_argument(
      "--device", default="cpu", type=str, help='use "cpu" or "gpu".'
  )
  args = parser.parse_args()

  numpyro.set_platform(args.device)
  coix.set_backend("coix.oryx")

  main(args)
