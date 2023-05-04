import argparse
from functools import partial
import sys

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import numpy as np

import flax
import flax.linen as nn
import jax
from jax import random
import jax.numpy as jnp
import optax

import coix
import numpyro
import numpyro.distributions as dist


### Networks

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
        beta_raw[0] + jnp.pad(jnp.cumsum(nn.softplus(beta_raw[1:])), (1, 0)))
    beta = jnp.pad(beta, (1, 1), constant_values=(0, 1))
    beta_k = beta[index]

    angles = 2 * jnp.arange(1, self.M + 1) * jnp.pi / self.M
    mu = 10 * jnp.stack([jnp.sin(angles), jnp.cos(angles)], -1)
    sigma = jnp.sqrt(0.5)
    target_density = nn.logsumexp(dist.Normal(mu, sigma).log_prob(x[..., None, :]).sum(-1), -1)
    init_proposal = dist.Normal(0, 5).log_prob(x).sum(-1)
    return beta_k * target_density + (1 - beta_k) * init_proposal


class AnnealKernelList(nn.Module):
  M = 8

  @nn.compact
  def __call__(self, x, index=0):
    if self.is_mutable_collection('params'):
      vmap_net = nn.vmap(
          AnnealKernel, variable_axes={'params': 0}, split_rngs={'params': True})
      out = vmap_net(name='kernel')(jnp.broadcast_to(x, (self.M - 1,) + x.shape))
      return jax.tree_util.tree_map(lambda x: x[index], out)
    params = self.scope.get_variable('params', 'kernel')
    params_i = jax.tree_util.tree_map(lambda x: x[index], params)
    return AnnealKernel(name='kernel').apply(flax.core.freeze({"params": params_i}), x)


class AnnealNetwork(nn.Module):

  def setup(self):
    self.forward_kernels = AnnealKernelList()
    self.reverse_kernels = AnnealKernelList()
    self.anneal_density = AnnealDensity()

  def __call__(self, x):
    self.reverse_kernels(x)
    self.anneal_density(x)
    return self.forward_kernels(x)


### Model and kernels

def anneal_target(network, k=0):
  x = numpyro.sample("x", dist.Normal(0, 5).expand([2]).mask(False).to_event())
  anneal_density = network.anneal_density(x, index=k)
  numpyro.sample("anneal_density", dist.Unit(anneal_density))
  return {"x": x},


def anneal_forward(network, inputs, k=0):
  mu, sigma = network.forward_kernels(inputs["x"], index=k)
  return numpyro.sample("x", dist.Normal(mu, sigma).to_event(1))


def anneal_reverse(network, inputs, k=0):
  mu, sigma = network.reverse_kernels(inputs["x"], index=k)
  return numpyro.sample("x", dist.Normal(mu, sigma).to_event(1))


### Train

def make_anneal(params, unroll=False, num_particles=10):
  network = coix.util.BindModule(AnnealNetwork(), params)
  # Add particle dimension and construct a program.
  make_particle_plate = lambda: numpyro.plate("particle", num_particles, dim=-1)
  targets = lambda k: make_particle_plate()(partial(anneal_target, network, k=k))
  forwards = lambda k: make_particle_plate()(partial(anneal_forward, network, k=k))
  reverses = lambda k: make_particle_plate()(partial(anneal_reverse, network, k=k))
  if unroll:  # to unroll the algorithm, we provide a list of programs
    targets = [targets(k) for k in range(8)]
    forwards = [forwards(k) for k in range(7)]
    reverses = [reverses(k) for k in range(7)]
  program = coix.algo.nvi_rkl(targets, forwards, reverses, num_targets=8)
  return program


def loss_fn(params, key, num_particles, unroll=False):
  # Run the program and get metrics.
  program = make_anneal(params, num_particles=num_particles, unroll=unroll)
  _, _, metrics = coix.traced_evaluate(program, seed=key)()
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
    init_params, optax.adam(lr), num_steps, jit_compile=True)

  rng_keys = random.split(random.PRNGKey(1), 100)

  def eval_program(seed):
    p = make_anneal(anneal_params, unroll=True, num_particles=1000)
    out, trace, metrics = coix.traced_evaluate(p, seed=seed)()
    return out, trace, metrics

  _, trace, metrics = jax.vmap(eval_program)(rng_keys)

  metrics.pop("log_weight")
  anneal_metrics = jax.tree_util.tree_map(lambda x: round(float(jnp.mean(x)), 4), metrics)
  print(anneal_metrics)

  plt.figure(figsize=(8, 8))
  x = trace["x"]["value"].reshape((-1, 2))
  H, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], bins=100)
  plt.imshow(H.T)
  plt.show()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Annealing example")
  parser.add_argument("--num_particles", nargs="?", default=36, type=int)
  parser.add_argument("--learning-rate", nargs="?", default=1e-3, type=float)
  parser.add_argument("--num-steps", nargs="?", default=20000, type=int)
  parser.add_argument("--unroll-loop", action="store_true")
  parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
  args = parser.parse_args()

  numpyro.set_platform(args.device)
  coix.set_backend("coix.numpyro")

  main(args)
