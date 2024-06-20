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

"""Program primitives and transforms."""

import functools
import inspect
import itertools

from coix.util import get_batch_ndims
from coix.util import get_log_weight
from coix.util import get_site_log_prob
import jax
import jax.numpy as jnp

from oryx.core import ppl
from oryx.core import primitive
from oryx.core import trace_util
from oryx.core.interpreters import harvest
from oryx.core.ppl import effect_handler
from oryx.distributions import distribution_extensions

random_variable_p = distribution_extensions.random_variable_p

__all__ = [
    "detach",
    "empirical",
    "factor",
    "prng_key",
    "rv",
    "stick_the_landing",
    "suffix",
    "traced_evaluate",
]

DISTRIBUTION = "distribution"
METRIC = "metric"
OBSERVED = "observed"
RANDOM_VARIABLE = ppl.RANDOM_VARIABLE

ALL_TAGS = (RANDOM_VARIABLE, OBSERVED, DISTRIBUTION, METRIC)

########################################
# Override Oryx behaviors
########################################


# Patch Oryx behavior to handle custom_jvp properly.
def _process_custom_jvp_call(
    self, trace, prim, fun, jvp, tracers, *, symbolic_zeros
):
  """Patch harvest.ReapContext.process_custom_jvp_call."""
  del self
  vals_in = [t.val for t in tracers]
  out_flat = prim.bind(fun, jvp, *vals_in, symbolic_zeros=symbolic_zeros)
  out_tracer = jax.util.safe_map(trace.pure, out_flat)
  return out_tracer


harvest.ReapContext.process_custom_jvp_call = _process_custom_jvp_call


def _eval_jaxpr_with_state(jaxpr, rules, consts, state, *args):
  """Patch effect_handler.eval_jaxpr_with_state."""
  env = effect_handler.Environment()

  jax.util.safe_map(env.write, jaxpr.constvars, consts)
  jax.util.safe_map(env.write, jaxpr.invars, args)

  for eqn in jaxpr.eqns:
    invals = jax.util.safe_map(env.read, eqn.invars)
    call_jaxpr, params = trace_util.extract_call_jaxpr(
        eqn.primitive, eqn.params
    )
    if eqn.primitive.name == "custom_jvp_call":
      subfuns, bind_params = eqn.primitive.get_bind_params(params)
      ans = eqn.primitive.bind(*subfuns, *invals, **bind_params)
    elif call_jaxpr:
      call_rule = effect_handler._effect_handler_call_rules.get(  # pylint: disable=protected-access
          eqn.primitive,
          functools.partial(
              effect_handler.default_call_interpreter_rule, eqn.primitive
          ),
      )
      ans, state = call_rule(rules, state, invals, call_jaxpr, **params)
    elif eqn.primitive in rules:
      ans, state = rules[eqn.primitive](state, *invals, **params)
    else:
      ans = eqn.primitive.bind(*invals, **params)
    if eqn.primitive.multiple_results:
      jax.util.safe_map(env.write, eqn.outvars, ans)
    else:
      env.write(eqn.outvars[0], ans)
  return jax.util.safe_map(env.read, jaxpr.outvars), state


effect_handler.eval_jaxpr_with_state = _eval_jaxpr_with_state


def identity(value, dist):
  del dist
  return value


def _dist_sample(key, dist):
  if "seed" in inspect.getfullargspec(dist.sample).args:
    return dist.sample(seed=key)
  else:
    return dist.sample(key)


def rv(dist, *, obs=None, name=None):
  """Declares a random variable."""

  # This behaves like oryx.core.ppl.rv but allows observed declaration
  # and batched distribution.

  def sample(key):
    if obs is None:
      sample_fn = _dist_sample
      sample_args = (key, dist)
    else:
      sample_fn = identity
      sample_args = (obs, dist)

    result = primitive.initial_style_bind(
        random_variable_p,
        batch_ndims=0,
        distribution_name=dist.__class__.__name__,
        name=name,
        mode="strict",
    )(sample_fn)(*sample_args)
    return harvest.sow(result, tag=RANDOM_VARIABLE, name=name)

  if obs is None:
    return sample
  else:
    return harvest.sow(sample(0), tag=OBSERVED, name=name)


@jax.tree_util.register_pytree_node_class
class Delta:
  """Dirac Delta distribution."""

  def __init__(self, value, log_density):
    self.value = value
    self.log_density = log_density

  def sample(self, key):
    del key
    return self.value

  def log_prob(self, value):
    del value
    return self.log_density

  def tree_flatten(self):
    return ((self.value, self.log_density), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


# We follow Pyro approach to use Unit distributions for factors.
@jax.tree_util.register_pytree_node_class
class Unit:
  """Unit Factor distribution."""

  def __init__(self, log_factor):
    self.log_factor = log_factor

  def sample(self, key):
    del key
    return jnp.empty((0,))

  def log_prob(self, value):
    del value
    return self.log_factor

  def tree_flatten(self):
    return ((self.log_factor,), None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    del aux_data
    return cls(*children)


def factor(log_factor, *, name=None):
  """Declare a factor to be added to a program."""
  return rv(Unit(log_factor), name=name)(0)


########################################
# Effect Handlers
########################################


def _split_list(args, num_consts):
  return jax.util.split_list(args, [num_consts])[1]


def substitute_rule(state, *args, **kwargs):
  """Rule for substitute handler."""
  name = kwargs.get("name")
  if name in state:
    flat_args = _split_list(args, kwargs["num_consts"])
    _, dist = jax.tree.unflatten(kwargs["in_tree"], flat_args)
    value = state[name]
    value = primitive.tie_in(flat_args, value)
    jaxpr, _ = trace_util.stage(identity, dynamic=True)(value, dist)
    kwargs["jaxpr"] = jaxpr.jaxpr
    kwargs["num_consts"] = len(jaxpr.literals)
    args = itertools.chain(jaxpr.literals, (value,), flat_args[1:])
  return random_variable_p.bind(*args, **kwargs), state


substitute_handler = ppl.make_effect_handler(
    {random_variable_p: substitute_rule}
)


def substitute(f, latents):
  """Runs `f` with latent values are obtained from `latents`."""

  def wrapped(*args, **kwargs):
    return substitute_handler(f)(latents, *args, **kwargs)[0]

  return wrapped


def distribution_rule(state, *args, **kwargs):
  """Rule for distribution handler."""
  name = kwargs.get("name")
  if name is not None:
    flat_args = _split_list(args, kwargs["num_consts"])
    _, dist = jax.tree.unflatten(kwargs["in_tree"], flat_args)
    dist_flat, dist_tree = jax.tree.flatten(dist)
    state[name] = {dist_tree: dist_flat}
  args = jax.tree.map(jax.core.raise_as_much_as_possible, args)
  return random_variable_p.bind(*args, **kwargs), state


distribution_handler = ppl.make_effect_handler(
    {random_variable_p: distribution_rule}
)


def tag_distribution(f):
  """Executes f with distributions tagged."""

  def wrapped(*args, **kwargs):
    out, fns = distribution_handler(f)({}, *args, **kwargs)
    for name, fn in fns.items():
      harvest.sow(fn, tag=DISTRIBUTION, name=name)
    return out

  return wrapped


def suffix_rule(state, *args, **kwargs):
  """Suffix rule for `sow_p` primitive."""
  if kwargs["tag"] in [OBSERVED, RANDOM_VARIABLE]:
    if kwargs["name"]:
      kwargs["name"] = kwargs["name"] + "_PREV_"
  return harvest.sow_p.bind(*args, **kwargs), state


def suffix_rv_rule(state, *args, **kwargs):
  """Suffix rule for `random_variable_p` primitive."""
  if kwargs.get("name"):
    kwargs["name"] = kwargs["name"] + "_PREV_"
  return random_variable_p.bind(*args, **kwargs), state


suffix_handler = ppl.make_effect_handler(
    {harvest.sow_p: suffix_rule, random_variable_p: suffix_rv_rule}
)


def suffix(f):
  """Adds suffix to random variables appeared in `names`."""

  def wrapped(*args, **kwargs):
    return suffix_handler(f)(None, *args, **kwargs)[0]

  return wrapped


def detach_rule(state, *args, **kwargs):
  """Rule for detach handler."""
  consts = args[: kwargs["num_consts"]]
  run_args = args[kwargs["num_consts"] :]

  def _run(*args):
    return jax.lax.stop_gradient(
        random_variable_p.bind(*itertools.chain(consts, args), **kwargs)
    )

  detach_jaxpr, _ = trace_util.stage(_run, dynamic=True)(*run_args)
  kwargs["jaxpr"] = detach_jaxpr.jaxpr
  return random_variable_p.bind(*args, **kwargs), state


detach_handler = ppl.make_effect_handler({random_variable_p: detach_rule})


def detach(f):
  """Detach handler."""

  def wrapped(*args, **kwargs):
    return detach_handler(f)(None, *args, **kwargs)[0]

  return wrapped


@jax.tree_util.register_pytree_node_class
class STLDistribution:
  """Sticking-the-landing log density."""

  def __init__(self, base_dist):
    self.base_dist = base_dist

  def sample(self, key):
    return _dist_sample(key, self.base_dist)

  def log_prob(self, value):
    return jax.lax.stop_gradient(self.base_dist).log_prob(value)

  def tree_flatten(self):
    params, treedef = jax.tree.flatten(self.base_dist)
    return (params, treedef)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    base_dist = jax.tree.unflatten(aux_data, children)
    return cls(base_dist)


def stl_rule(state, *args, **kwargs):
  flat_args = _split_list(args, kwargs["num_consts"])
  key, dist = jax.tree.unflatten(kwargs["in_tree"], flat_args)
  stl_dist = STLDistribution(dist)
  _, in_tree = jax.tree.flatten((key, stl_dist))
  kwargs["in_tree"] = in_tree
  out = random_variable_p.bind(*args, **kwargs)
  return out, state


stl_handler = ppl.make_effect_handler({random_variable_p: stl_rule})


def stick_the_landing(f):
  def wrapped(*args, **kwargs):
    return stl_handler(f)(None, *args, **kwargs)[0]

  return wrapped


########################################
# Reap helpers
########################################


def call_and_reap_tags(f, tags):
  """A helper to collect values from a sequence of tags."""
  tags = [tags] if isinstance(tags, str) else tags

  def wrapped(*args, **kwargs):
    f_with_tag = f
    for tag in tags:
      f_with_tag = harvest.call_and_reap(f_with_tag, tag=tag)
    out_with_tag = f_with_tag(*args, **kwargs)
    tags_dict = {}
    for tag in tags[::-1]:
      out_with_tag, values = out_with_tag
      tags_dict[tag] = values
    return out_with_tag, tags_dict

  return wrapped


def traced_evaluate(p, latents=None):
  """Perform traced evaluation.

  Args:
    p: a program
    latents: optional values to be substituted into `p`

  Returns:
    (out, p_trace, p_metrics): a tuple of marginal output, trace, and metrics
  """

  def wrapped(*args, **kwargs):
    p_subs = substitute(p, latents=latents) if latents is not None else p
    p_tagged = tag_distribution(p_subs)
    out, tags = call_and_reap_tags(p_tagged, ALL_TAGS)(*args, **kwargs)
    trace = {}
    for name, value in tags[RANDOM_VARIABLE].items():
      dist_tree, dist_flat = list(tags[DISTRIBUTION][name].items())[0]
      dist = jax.tree.unflatten(dist_tree, dist_flat)
      trace[name] = {"value": value, "log_prob": dist.log_prob(value)}
      if name in tags[OBSERVED]:
        trace[name]["is_observed"] = True
    metrics = tags[METRIC]
    if "loss" not in metrics:
      metrics["loss"] = jnp.array(0.0)
    if "log_density" not in metrics:
      log_density = sum(jnp.sum(site["log_prob"]) for site in trace.values())
      metrics["log_density"] = jnp.array(0.0) + log_density
    if "log_weight" not in metrics:
      log_probs = [get_site_log_prob(site) for site in trace.values()]
      weight = get_log_weight(trace, get_batch_ndims(log_probs))
      metrics = {**metrics, "log_weight": weight}
    return out, trace, metrics

  return wrapped


def sow_metric(value, name):
  return harvest.sow(value, tag=METRIC, name=name)


def empirical(out, trace, metrics):
  """Creates a deterministic program with Delta variables."""

  def wrapped(*args, **kwargs):
    tie_trace, tie_metrics = primitive.tie_in((args, kwargs), (trace, metrics))
    for name, site in tie_trace.items():
      value, lp = site["value"], site["log_prob"]
      if "is_observed" in site:
        rv(Delta(value, lp), obs=value, name=name)
      else:
        rv(Delta(value, lp), name=name)(0)
    for name, value in tie_metrics.items():
      sow_metric(value, name)
    return out

  return wrapped


def prng_key():
  raise ValueError("Cannot genenerate random key under the oryx backend.")
