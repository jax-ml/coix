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

"""Program combinators.

The implement is pretty much backend-agnostic. We just assume that the core
backend supports the following functionality:
  + `suffix(p)`: rename latent variables of the program `p`,
  + `traced_evaluate(p, latents=None)`: execute `p` and collect trace, metrics,
    optionally we can substitute values in `latents` to `p`,
  + `empirical(out, trace, metrics)`: create a delta program given output,
    trace, and metrics. Inputs of `empirical` are outputs of `traced_evaluate`.
"""

import functools

from coix import core
from coix import util
import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "compose",
    "extend",
    "fori_loop",
    "propose",
    "resample",
]


def compose(q2, q1, suffix=True):
  """Executes q2(*q1(...)).

  Note: We only allow at most one of `q1` or `q2` is weighted.

  Args:
    q2: a program
    q1: a program
    suffix: whether to add suffix `_PREV_` to variables in `q1`

  Returns:
    q: the composed program
  """

  def wrapped(*args, **kwargs):
    q = core.suffix(q1) if suffix else q1
    return q2(*q(*args, **kwargs))

  return wrapped


def extend(p, f):
  """Executes f(*p(...)) with random variables in f marked as auxiliary.

  Note: We don't allow recursively marginalize out `p` yet.

  Args:
    p: a target program
    f: an auxiliary program

  Returns:
    p_new: the extended program
  """

  def wrapped(*args, **kwargs):
    args = p(*args, **kwargs)
    core.suffix(f)(*args)
    return args

  return wrapped


def _get_batch_ndims(log_probs):
  if not log_probs:
    return 0
  min_ndim = min(jnp.ndim(lp) for lp in log_probs)
  batch_ndims = 0
  for i in range(min_ndim):
    if len(set(jnp.shape(lp)[i] for lp in log_probs)) > 1:
      break
    batch_ndims = batch_ndims + 1
  return batch_ndims


def _get_log_weight(trace, batch_ndims):
  """Computes log weight of the trace and keeps its batch dimensions."""
  log_weight = jnp.zeros((1,) * batch_ndims)
  for site in trace.values():
    lp = util.get_site_log_prob(site)
    if util.is_observed_site(site):
      log_weight = log_weight + jnp.sum(
          lp, axis=tuple(range(batch_ndims - jnp.ndim(lp), 0))
      )
    else:
      log_weight = log_weight + jnp.zeros(jnp.shape(lp)[:batch_ndims])
  return log_weight


def _split_key(key):
  keys = jax.vmap(jax.random.split)(key.reshape(-1, 2)).reshape(
      key.shape[:-1] + (2, 2)
  )
  return keys[..., 0, :], keys[..., 1, :]


def _fold_in_key(key, i):
  key_new = jax.vmap(jax.random.fold_in, (0, None))(key.reshape(-1, 2), i)
  return key_new.reshape(key.shape)


def propose(p, q, *, loss_fn=None, detach=False):
  """Returns a new program with important weight.

  We assume the leftmost batch dimension is the particle dimension. You can add
  additional batch dimensions to the whole program by using `vmap`, e.g.
  `vmap(propose(p, q))`.

  Note: We assume superfluous variables, which appear in `q` but not in `p`,
    implicitly follow Delta distribution in `p`.

  Args:
    p: a target program
    q: a proposal program
    loss_fn: a function that computes loss of this propose combinator
    detach: whether to detach `value` of the returned program

  Returns:
    q_new: the proposed program
  """

  def wrapped(*args, **kwargs):
    if util.can_extract_key(args):
      key_p, key_q = _split_key(args[0])
      p_args = (key_p,) + args[1:]
      q_args = (key_q,) + args[1:]
    else:
      p_args = q_args = args
    _, q_trace, q_metrics = core.traced_evaluate(q)(*q_args, **kwargs)
    metrics = q_metrics.copy()
    q_latents = {
        name: util.get_site_value(site)
        for name, site in q_trace.items()
        if not util.is_observed_site(site)
    }
    traced_p = core.traced_evaluate(p, latents=q_latents)
    out, p_trace, _ = traced_p(*p_args, **kwargs)

    p_log_probs = {
        name: util.get_site_log_prob(site) for name, site in p_trace.items()
    }
    q_log_probs = {
        name: util.get_site_log_prob(site) for name, site in q_trace.items()
    }
    log_probs = list(p_log_probs.values()) + list(q_log_probs.values())
    batch_ndims = _get_batch_ndims(log_probs)

    if "log_weight" in q_metrics:
      in_log_weight = q_metrics["log_weight"]
      in_log_weight = jnp.sum(
          in_log_weight,
          axis=tuple(range(batch_ndims - jnp.ndim(in_log_weight), 0)),
      )
    else:
      in_log_weight = _get_log_weight(q_trace, batch_ndims)
    p_log_weight = sum(
        lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
        for name, lp in p_log_probs.items()
        if util.is_observed_site(p_trace[name]) or (name in q_trace)
    )
    # Note: We include superfluous variables, whose `name in p_trace`.
    q_log_weight = sum(
        lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
        for name, lp in q_log_probs.items()
    )
    incremental_log_weight = p_log_weight - q_log_weight
    log_weight = in_log_weight + incremental_log_weight
    metrics["log_weight"] = log_weight

    if batch_ndims:  # leftmost dimension is particle dimension
      ess = 1 / (jax.nn.softmax(log_weight, axis=0) ** 2).sum(0)
      metrics["ess"] = ess.mean()
      log_z = jax.scipy.special.logsumexp(log_weight, 0) - jnp.log(
          log_weight.shape[0]
      )
      metrics["log_Z"] = log_z.sum()

    if loss_fn is not None:
      if detach:
        p_latents = {
            name: util.get_site_value(site, detach=True)
            for name, site in p_trace.items()
            if not util.is_observed_site(site)
        }
        out, p_trace, _ = core.traced_evaluate(p, latents=p_latents)(
            *p_args, **kwargs
        )
      loss = loss_fn(q_trace, p_trace, in_log_weight, incremental_log_weight)
      metrics["loss"] = q_metrics.get("loss", 0.0) + loss

    marginal_trace = {
        name: site
        for name, site in p_trace.items()
        if not name.endswith("_PREV_")
    }
    log_density = jnp.zeros((1,) * batch_ndims) + sum(
        lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
        for name, lp in p_log_probs.items()
        if name in marginal_trace
    )
    if batch_ndims:
      log_density = jnp.mean(log_density, axis=0).sum()
    metrics["log_density"] = log_density
    return core.empirical(out, marginal_trace, metrics)(*args, **kwargs)

  return wrapped


def _maybe_get_along_first_axis(x, idx, n, squeeze=False):
  """Get along the first axis of `x` if `x.shape[0] == n`."""
  is_list = False
  if isinstance(x, list):
    is_list = True
    x = np.array(x)
  # Special treatment for cascades.
  if hasattr(x, "value"):
    x.value = _maybe_get_along_first_axis(
        util.get_site_value(x), idx, n, squeeze=squeeze
    )
  if hasattr(x, "log_density"):
    x.log_density = _maybe_get_along_first_axis(
        util.get_site_log_prob(x), idx, n, squeeze=squeeze
    )
  if (
      isinstance(x, (np.ndarray, jnp.ndarray))
      and (x.ndim >= 1)
      and (x.shape[0] == n)
  ):
    idx = idx.reshape(idx.shape + (1,) * (x.ndim - idx.ndim))
    if isinstance(x, np.ndarray):
      y = np.take_along_axis(x, idx, axis=0)
    else:
      y = jnp.take_along_axis(x, idx, axis=0)
    y = y.tolist() if is_list else y
    return y[0] if (idx.shape[0] == 1 and squeeze) else y
  else:
    return x


def resample(q, num_samples=None):
  """Returns a new program with equally-weighted particles.

  Args:
    q: a program
    num_samples: the number of samples after resampling. Set this to an empty
      tuple to draw 1 sample without the leftmost singleton dimension. Defaults
      to the number of particles in `q`.

  Returns:
    q_new: the resampled program
  """

  def fn(*args, **kwargs):
    if util.can_extract_key(args):
      key_r, key_q = _split_key(args[0])
      # We just need a single key for resampling.
      key_r = key_r.reshape((-1, 2)).sum(0)
      args = (key_q,) + args[1:]
    else:
      key_r = core.prng_key()
    out, trace, q_metrics = core.traced_evaluate(q)(*args, **kwargs)
    log_probs = {
        name: util.get_site_log_prob(site) for name, site in trace.items()
    }
    batch_ndims = _get_batch_ndims(log_probs.values())
    weighted = ("log_weight" in q_metrics) or any(
        util.is_observed_site(site) for site in trace.values()
    )
    if (batch_ndims == 0) or not weighted:  # resample is no-op
      return core.empirical(out, trace, q_metrics)(*args, **kwargs)

    metrics = q_metrics.copy()
    if "log_weight" in q_metrics:
      in_log_weight = q_metrics.pop("log_weight")
      in_log_weight = jnp.sum(
          in_log_weight,
          axis=tuple(range(batch_ndims - jnp.ndim(in_log_weight), 0)),
      )
    else:
      in_log_weight = _get_log_weight(trace, batch_ndims)
    n = in_log_weight.shape[0]
    k = n if num_samples is None else num_samples
    log_weight = jax.nn.logsumexp(in_log_weight, 0) - jnp.log(k if k else 1)
    if k:
      metrics["log_weight"] = jnp.broadcast_to(
          log_weight, (k,) + in_log_weight.shape[1:]
      )
      metrics["ess"] = jnp.asarray(float(k))
    if "log_Z" not in q_metrics:
      metrics["log_Z"] = log_weight.sum()

    log_probs = jax.nn.log_softmax(in_log_weight, axis=0)
    idx = util.get_systematic_resampling_indices(
        log_probs, rng_key=key_r, num_samples=k if k else 1
    )
    maybe_get_along_first_axis = functools.partial(
        _maybe_get_along_first_axis, idx=idx, n=n, squeeze=not k
    )
    out = jax.tree_util.tree_map(
        maybe_get_along_first_axis, out, is_leaf=lambda x: isinstance(x, list)
    )
    resample_trace = jax.tree_util.tree_map(
        maybe_get_along_first_axis, trace, is_leaf=lambda x: isinstance(x, list)
    )
    return core.empirical(out, resample_trace, metrics)(*args, **kwargs)

  return fn


def _add_missing_metrics(metrics, trace):
  """Adds missing metrics to get consistent pytree in fori_loop."""
  full_metrics = metrics.copy()
  log_probs = {
      name: util.get_site_log_prob(site) for name, site in trace.items()
  }
  if "log_weight" not in metrics:
    batch_ndims = min(_get_batch_ndims(list(log_probs.values())), 1)
    log_weight = _get_log_weight(trace, batch_ndims)
    full_metrics["log_weight"] = log_weight
    if batch_ndims:  # leftmost dimension is particle dimension
      ess = 1 / (jax.nn.softmax(log_weight, axis=0) ** 2).sum(0)
      full_metrics["ess"] = ess.mean()
      n = log_weight.shape[0]
      log_z = jax.scipy.special.logsumexp(log_weight, 0) - jnp.log(n)
      full_metrics["log_Z"] = log_z.mean()
  if "loss" not in metrics:
    full_metrics["loss"] = jnp.array(0.0)
  if "log_density" not in metrics:
    log_density = sum(jnp.sum(lp) for lp in log_probs.values())
    full_metrics["log_density"] = jnp.array(0.0) + log_density
  return full_metrics


def fori_loop(lower, upper, body_fun, init_program):
  """Returns a program which loops over programs created by body_fun.

  Args:
    lower: loop index lower bound
    upper: loop index upper bound (exclusive)
    body_fun: a function that takes a pair of inputs (index, program) and return
      a new program
    init_program: initial program for `body_fun`

  Returns:
    q: the final program
  """

  def fn(*args, **kwargs):
    if util.can_extract_key(args):
      key = args[0]

      def trace_fn(fn, key):
        return core.traced_evaluate(fn)(key, *args[1:], **kwargs)

    else:
      key = core.prng_key()

      def trace_fn(fn, key):
        return core.traced_evaluate(fn, seed=key)(*args, **kwargs)

    key_body, key_init = _split_key(key)

    def jax_body_fun(i, val):
      q = core.empirical(*val)
      return trace_fn(body_fun(i, q), _fold_in_key(key_body, i))

    v, trace, metrics = trace_fn(init_program, key_init)
    metrics = _add_missing_metrics(metrics, trace)
    output = jax.lax.fori_loop(lower, upper, jax_body_fun, (v, trace, metrics))
    return core.empirical(*output)(key, *args, **kwargs)

  return fn


def _join_samples(first_set, second_set):
  if first_set is None:
    return second_set
  if isinstance(first_set, list):
    assert isinstance(second_set, list)
    return first_set + second_set
  else:
    return jnp.concatenate([first_set, second_set], axis=0)


def memoize(p, q, memory=None, memory_size=None):
  """Returns a new program using additional samples from memory for proposal.

  Args:
    p: a target program
    q: a proposal program
    memory: additional samples for `q`
    memory_size: size of the memory to be stored in the program's metrics

  Returns:
    q_new: the proposed program
  """
  if (memory is None) and (memory_size is None):
    raise ValueError("One of memory or memory_size needs to be specified.")
  memory = {} if memory is None else memory
  memory_sizes = [len(x) for x in memory.values()]
  if len(set(memory_sizes)) > 1:
    raise ValueError("We need all variables have the same memory size.")
  memory_size = memory_size if memory_size is not None else memory_sizes[0]

  def wrapped(*args, **kwargs):
    if util.can_extract_key(args):
      key = args[0]
      p_key, q_key = key + jnp.asarray([1, 0], dtype=key.dtype), key + 1
      p_args = (p_key,) + args[1:]
      q_args = (q_key,) + args[1:]
    else:
      p_args = q_args = args
    _, q_trace, q_metrics = core.traced_evaluate(q)(*q_args, **kwargs)
    metrics = q_metrics.copy()
    q_latents = {
        name: _join_samples(memory.get(name, None), util.get_site_value(site))
        for name, site in q_trace.items()
        if not util.is_observed_site(site)
    }
    traced_p = core.traced_evaluate(p, latents=q_latents)
    out, p_trace, _ = traced_p(*p_args, **kwargs)

    p_log_probs = {
        name: util.get_site_log_prob(site) for name, site in p_trace.items()
    }
    batch_ndims = _get_batch_ndims(p_log_probs.values())

    p_log_weight = sum(
        lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
        for name, lp in p_log_probs.items()
    )

    marginal_trace = {
        name: site
        for name, site in p_trace.items()
        if not util.is_observed_site(site)
    }
    new_memory = {
        name: util.get_site_value(site) for name, site in marginal_trace.items()
    }
    num_particles = p_log_weight.shape[0]
    batch_dim = p_log_weight.ndim
    flat_memory = {
        k: np.array(v) if isinstance(v, list) else v
        for k, v in new_memory.items()
    }
    flat_memory = {
        k: v.reshape((num_particles, -1) + v.shape[batch_dim:])
        for k, v in flat_memory.items()
    }
    flat_log_weight = p_log_weight.reshape(p_log_weight.shape[:1] + (-1,))
    idxs = []
    for i in range(flat_log_weight.shape[1]):
      w = flat_log_weight[:, i]
      mem = [
          [flat_memory[k][j, i] for k in sorted(flat_memory)]
          for j in range(num_particles)
      ]
      unique_idx = np.unique(mem, axis=0, return_index=True)[1]
      sorted_idx = jnp.argsort(w[unique_idx])[::-1][:memory_size]
      idxs.append(unique_idx[sorted_idx])
    idxs = jnp.stack(idxs, -1).reshape((memory_size,) + p_log_weight.shape[1:])

    maybe_get_along_first_axis = functools.partial(
        _maybe_get_along_first_axis, idx=idxs, n=num_particles
    )
    metrics["log_weight"] = maybe_get_along_first_axis(p_log_weight)
    out = jax.tree_util.tree_map(
        maybe_get_along_first_axis, out, is_leaf=lambda x: isinstance(x, list)
    )
    marginal_trace = jax.tree_util.tree_map(
        maybe_get_along_first_axis,
        marginal_trace,
        is_leaf=lambda x: isinstance(x, list),
    )
    metrics["memory"] = jax.tree_util.tree_map(
        maybe_get_along_first_axis,
        new_memory,
        is_leaf=lambda x: isinstance(x, list),
    )
    return core.empirical(out, marginal_trace, metrics)(*args, **kwargs)

  return wrapped
