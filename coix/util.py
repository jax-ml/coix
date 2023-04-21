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

"""Utilities."""

import functools
import time

import jax
from jax import random
import jax.numpy as jnp
import numpy as np


def get_systematic_resampling_indices(log_weights, rng_key, num_samples):
  """Gets resampling indices based on systematic resampling."""
  n = log_weights.shape[0]
  weight = jax.nn.softmax(log_weights, axis=0)
  cummulative_weight = weight.cumsum(axis=0)
  cummulative_weight = cummulative_weight / cummulative_weight[-1]
  cummulative_weight = cummulative_weight.reshape((n, -1)).swapaxes(0, 1)
  m = cummulative_weight.shape[0]
  if rng_key is not None:
    uniform = jax.random.uniform(rng_key, (m,))
  else:
    uniform = np.random.rand(m)
  positions = (uniform[:, None] + np.arange(num_samples)) / num_samples
  shift = np.arange(m)[:, None]
  cummulative_weight = (cummulative_weight + 2 * shift).reshape(-1)
  positions = (positions + 2 * shift).reshape(-1)
  index = cummulative_weight.searchsorted(positions)
  index = (index.reshape(m, num_samples) - n * shift).swapaxes(0, 1)
  return index.reshape((num_samples,) + log_weights.shape[1:])


def get_site_log_prob(site):
  if hasattr(site, "log_density"):
    return site.log_density
  else:
    return site["log_prob"]


def get_site_value(site, detach=False):
  if hasattr(site, "value"):
    value = site.value
  else:
    value = site["value"]
  if detach and isinstance(value, jnp.ndarray):
    return jax.lax.stop_gradient(value)
  else:
    return value


def is_observed_site(site):
  if hasattr(site, "tag"):
    return site.tag == "observed"
  else:
    return "is_observed" in site


def can_extract_key(args):
  return (
      args
      and isinstance(args[0], jnp.ndarray)
      and (args[0].dtype == jnp.uint32)
      and (jnp.ndim(args[0]) >= 1)
      and (args[0].shape[-1] == 2)
  )


class _ChildModule:
  """A child of a bind module."""

  def __init__(self, module, params, name):
    self.module = module
    self.params = params
    self.name = name

  def __getitem__(self, i):
    return functools.partial(
        self.module.apply,
        self.params,
        method=lambda n, *a, **kw: getattr(n, self.name)[i](*a, **kw),
    )

  def __call__(self, *args, **kwargs):
    return self.module.apply(
        self.params,
        *args,
        method=lambda n, *a, **kw: getattr(n, self.name)(*a, **kw),
        **kwargs,
    )


class BindModule:
  """Like Flax's `module.bind(params)` but composed with JAX transforms."""

  def __init__(self, module, params):
    self.module = module
    self.params = params
    for submodule in params["params"]:
      setattr(
          self, submodule, _ChildModule(self.module, self.params, submodule)
      )
    for submodule in params["params"]:
      if "_" in submodule and submodule.split("_")[-1].isnumeric():
        maybe_submodule_list = "_".join(submodule.split("_")[:-1])
        if not hasattr(self, maybe_submodule_list):
          setattr(
              self,
              maybe_submodule_list,
              _ChildModule(self.module, self.params, maybe_submodule_list),
          )
    for field in module.__annotations__:
      if field not in ("parent", "name"):
        setattr(self, field, getattr(module, field))

  def __call__(self, *args, **kwargs):
    return self.module.apply(self.params, *args, **kwargs)


def train(
    loss_fn,
    init_params,
    optimizer,
    num_steps,
    dataloader=None,
    seed=0,
    jit_compile=True,
    eval_fn=None,
    log_every=None,
    **kwargs,
):
  """Optimize the parameters."""

  def step_fn(params, opt_state, *args, **kwargs):
    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, *args, **kwargs
    )
    grads = jax.tree_util.tree_map(
        lambda x, y: x.astype(y.dtype), grads, params)
    zeros_like = lambda g, o, p: (jax.tree_util.tree_map(jnp.zeros_like, g), o)
    updates, opt_state = jax.lax.cond(
        jnp.isfinite(jax.flatten_util.ravel_pytree(grads)[0]).all(),
        optimizer.update,
        zeros_like,
        grads,
        opt_state,
        params,
    )
    params = jax.tree_util.tree_map(lambda p, u: p + u, params, updates)
    return params, opt_state, metrics

  if callable(jit_compile):
    maybe_jitted_step_fn = jit_compile(step_fn)
  else:
    maybe_jitted_step_fn = jax.jit(step_fn) if jit_compile else step_fn
  opt_state = optimizer.init(init_params)
  params = init_params
  run_key = random.PRNGKey(seed)
  log_every = max(num_steps // 20, 1) if log_every is None else log_every
  space = str(len(str(num_steps - 1)))
  kwargs = kwargs.copy()
  if eval_fn is not None:
    print("Evaluating with the initial params...", flush=True)
    tic = time.time()
    eval_fn(0, params, **kwargs)
    print("Time to compile an eval step:", time.time() - tic,
          flush=True)
  print("Compiling the first train step...", flush=True)
  tic = time.time()
  for step in range(1, num_steps + 1):
    key = random.fold_in(run_key, step)
    args = (key, next(dataloader)) if dataloader is not None else (key,)
    params, opt_state, metrics = maybe_jitted_step_fn(
        params, opt_state, *args, **kwargs
    )
    for name, value in kwargs.items():
      if name in metrics:
        kwargs[name] = metrics[name]
    if step == 1:
      print("Time to compile a train step:", time.time() - tic,
            flush=True)
      print("=====", flush=True)
    if (step == num_steps) or (step % log_every == 0):
      log = ("Step {:<" + space + "d}").format(step)
      for name, value in sorted(metrics.items()):
        if np.isscalar(value) or (
            isinstance(value, (np.ndarray, jnp.ndarray)) and (value.ndim == 0)
        ):
          log += " | {} {:10.4f}".format(name, value)
      print(log, flush=True)
      if eval_fn is not None:
        eval_fn(step, params, **kwargs)
  return params, metrics
