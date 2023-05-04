import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers

prng_key = numpyro.prng_key


def traced_evaluate(p, latents=None, seed=None):
  def wrapped(*args, **kwargs):
    data = {} if latents is None else latents
    rng_seed = numpyro.prng_key() if seed is None else seed
    subs_model = handlers.seed(handlers.substitute(p, data=data), rng_seed=rng_seed)
    with handlers.block(), handlers.trace() as tr:
      out = subs_model(*args, **kwargs)
    trace = {}
    for name, site in tr.items():
      if site["type"] == "sample":
        value = site["value"]
        log_prob = site["fn"].log_prob(value)
        trace[name] = {"value": value, "log_prob": log_prob}
        if site.get("is_observed", False):
          trace[name]["is_observed"] = True
    metrics = {name: site["value"] for name, site in tr.items()
               if site["type"] == "metric"}
    return out, trace, metrics

  return wrapped


def add_metric(name, value):
  if numpyro.primitives._PYRO_STACK:
    msg = {"type": "metric", "value": value, "name": name}
    numpyro.primitives.apply_stack(msg)


def empirical(out, trace, metrics):
  def wrapped(*args, **kwargs):
    for name, site in trace.items():
      value, lp = site["value"], site["log_prob"]
      event_dim = jnp.ndim(value) - jnp.ndim(lp)
      obs = value if "is_observed" in site else None
      numpyro.sample(name, dist.Delta(value, lp, event_dim=event_dim), obs=obs)
    for name, value in metrics.items():
      add_metric(name, value)
    return out

  return wrapped


class suffix(numpyro.primitives.Messenger):
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
    d = jax.lax.stop_gradient(self.base_dist) if self.detach_args else self.base_dist
    return d.log_prob(value)

  def tree_flatten(self):
    params, treedef = jax.tree_util.tree_flatten(self.base_dist)
    return params, (treedef, self.detach_sample, self.detach_args)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
    treedef, detach_sample, detach_args = aux_data
    base_dist = jax.tree_util.tree_unflatten(treedef, children)
    return cls(base_dist, detach_sample=detach_sample, detach_args=detach_args)


class detach(numpyro.primitives.Messenger):
  def process_message(self, msg):
    if msg["type"] == "sample" and not msg.get("is_observed", False):
      msg["fn"] = StopGradient(msg["fn"], detach_sample=True)


class stick_the_landing(numpyro.primitives.Messenger):
  def process_message(self, msg):
    if msg["type"] == "sample" and not msg.get("is_observed", False):
      msg["fn"] = StopGradient(msg["fn"], detach_args=True)

