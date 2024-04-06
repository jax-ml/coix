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

"""Backend implementation for NumPyro."""

from coix.util import get_batch_ndims
from coix.util import get_log_weight
from coix.util import get_site_log_prob
import jax
import jax.numpy as jnp
import numpyro
from numpyro import handlers
import numpyro.distributions as dist

__all__ = [
    "detach",
    "empirical",
    "prng_key",
    "stick_the_landing",
    "suffix",
    "traced_evaluate",
]

prng_key = numpyro.prng_key


def traced_evaluate(p, latents=None, seed=None):
  """Performs traced evaluation for a program `p`."""

  def wrapped(*args, **kwargs):
    data = {} if latents is None else latents
    rng_seed = numpyro.prng_key() if seed is None else seed
    subs_model = handlers.seed(
        handlers.substitute(p, data=data), rng_seed=rng_seed
    )
    with handlers.block(), handlers.trace() as tr:
      out = subs_model(*args, **kwargs)
    trace = {}
    for name, site in tr.items():
      if site["type"] == "sample":
        value = site["value"]
        log_prob = site["fn"].log_prob(value)
        event_dim_holder = jnp.empty([1] * site["fn"].event_dim)
        trace[name] = {
            "value": value,
            "log_prob": log_prob,
            "_event_dim_holder": event_dim_holder,
        }
        if site.get("is_observed", False):
          trace[name]["is_observed"] = True
    metrics = {
        name: site["value"]
        for name, site in tr.items()
        if site["type"] == "metric"
    }
    # add log_weight to metrics
    if "log_weight" not in metrics:
      log_probs = [get_site_log_prob(site) for site in trace.values()]
      weight = get_log_weight(trace, get_batch_ndims(log_probs))
      metrics = {**metrics, "log_weight": weight}
    return out, trace, metrics

  return wrapped


def add_metric(name, value):
  """A NumPyro primitive to add `metric` type to a program."""
  if numpyro.primitives._PYRO_STACK:  # pylint:disable=protected-access
    msg = {"type": "metric", "value": value, "name": name}
    numpyro.primitives.apply_stack(msg)


def empirical(out, trace, metrics):
  """A program that produces `out`, `trace`, and `metrics` under evaluation."""

  def wrapped(*args, **kwargs):
    del args, kwargs
    for name, site in trace.items():
      value, lp = site["value"], site["log_prob"]
      event_dim = jnp.ndim(site["_event_dim_holder"])
      obs = value if "is_observed" in site else None
      numpyro.sample(name, dist.Delta(value, lp, event_dim=event_dim), obs=obs)
    for name, value in metrics.items():
      add_metric(name, value)
    return out

  return wrapped


class suffix(numpyro.primitives.Messenger):  # pylint:disable=invalid-name

  def process_message(self, msg):
    if msg["type"] == "sample":
      msg["name"] = msg["name"] + "_PREV_"


class StopGradient(dist.Distribution):
  """Nonreparameterized or stick-the-landing distribution."""

  def __init__(self, base_dist, detach_sample=False, detach_args=False):
    self.base_dist = base_dist
    self.detach_sample, self.detach_args = detach_sample, detach_args
    super().__init__(base_dist.batch_shape, base_dist.event_shape)

  def sample(self, key, sample_shape=()):
    samples = self.base_dist.sample(key, sample_shape=sample_shape)
    return jax.lax.stop_gradient(samples) if self.detach_sample else samples

  def log_prob(self, value):
    d = (
        jax.lax.stop_gradient(self.base_dist)
        if self.detach_args
        else self.base_dist
    )
    return d.log_prob(value)

  def tree_flatten(self):
    params, treedef = jax.tree_util.tree_flatten(self.base_dist)
    return params, (treedef, self.detach_sample, self.detach_args)

  @classmethod
  def tree_unflatten(cls, aux_data, params):
    treedef, detach_sample, detach_args = aux_data
    base_dist = jax.tree_util.tree_unflatten(treedef, params)
    return cls(base_dist, detach_sample=detach_sample, detach_args=detach_args)


class detach(numpyro.primitives.Messenger):  # pylint:disable=invalid-name

  def process_message(self, msg):
    if msg["type"] == "sample" and not msg.get("is_observed", False):
      msg["fn"] = StopGradient(msg["fn"], detach_sample=True)


class stick_the_landing(numpyro.primitives.Messenger):  # pylint:disable=invalid-name

  def process_message(self, msg):
    if msg["type"] == "sample" and not msg.get("is_observed", False):
      msg["fn"] = StopGradient(msg["fn"], detach_args=True)
