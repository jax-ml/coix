"""Program transforms."""

import importlib

__all__ = [
    "detach",
    "empirical",
    "factor",
    "get_backend_name",
    "prng_key",
    "rv",
    "stick_the_landing",
    "suffix",
    "register_backend",
    "set_backend",
    "traced_evaluate",
]

_BACKENDS = {}
_COIX_BACKEND = None


# pylint:disable=redefined-outer-name
def register_backend(
    backend,
    traced_evaluate=None,
    empirical=None,
    suffix=None,
    detach=None,
    stick_the_landing=None,
    rv=None,
    factor=None,
    prng_key=None,
):
  """Register backend."""
  fn_map = {
      "traced_evaluate": traced_evaluate,
      "empirical": empirical,
      "suffix": suffix,
      "detach": detach,
      "stick_the_landing": stick_the_landing,
      "rv": rv,
      "factor": factor,
      "prng_key": prng_key,
  }
  _BACKENDS[backend] = fn_map


# pylint:enable=redefined-outer-name


def set_backend(backend):
  """Set backend."""
  global _COIX_BACKEND

  if backend not in _BACKENDS:
    module = importlib.import_module(backend)
    fn_map = {}
    for fn in [
        "traced_evaluate",
        "empirical",
        "suffix",
        "detach",
        "stick_the_landing",
        "rv",
        "factor",
        "prng_key",
    ]:
      fn_map[fn] = getattr(module, fn, None)
    register_backend(backend, **fn_map)

  _COIX_BACKEND = backend


def get_backend_name():
  return _COIX_BACKEND


def get_backend():
  backend = _COIX_BACKEND
  if backend is None:
    set_backend("coix.oryx")
    return _BACKENDS["coix.oryx"]
  else:
    return _BACKENDS[backend]


########################################
# Program transforms
########################################


def _remove_suffix(name):
  i = 0
  while name.endswith("_PREV_"):
    i += len("_PREV_")
    name = name[: -len("_PREV_")]
  return name, i


def desuffix(trace):
  """Remove unnecessary suffix terms added to the trace."""
  names_to_raw_names = {}
  num_suffix_min = {}
  for name in trace:
    raw_name, num_suffix = _remove_suffix(name)
    names_to_raw_names[name] = raw_name
    if raw_name in num_suffix_min:
      num_suffix_min[raw_name] = min(num_suffix_min[raw_name], num_suffix)
    else:
      num_suffix_min[raw_name] = num_suffix
  new_trace = {}
  for name in trace:
    raw_name = names_to_raw_names[name]
    new_trace[name[: len(name) - num_suffix_min[raw_name]]] = trace[name]
  return new_trace


def traced_evaluate(p, latents=None, rng_seed=None, **kwargs):
  """Performs traced evaluation for a program `p`."""
  kwargs = kwargs.copy()
  if rng_seed is not None:
    kwargs["rng_seed"] = rng_seed
  fn = get_backend()["traced_evaluate"](p, latents=latents, **kwargs)

  def wrapped(*args, **kwargs):
    out, trace, metrics = fn(*args, **kwargs)
    return out, desuffix(trace), metrics

  return wrapped


def empirical(out, trace, metrics):
  return get_backend()["empirical"](out, trace, metrics)


def suffix(p):
  fn = get_backend()["suffix"]
  if fn is not None:
    return fn(p)
  else:
    return p


def detach(p):
  fn = get_backend()["detach"]
  if fn is not None:
    return fn(p)
  else:
    return p


def stick_the_landing(p):
  fn = get_backend()["stick_the_landing"]
  if fn is not None:
    return fn(p)
  else:
    return p


def rv(*args, **kwargs):
  fn = get_backend()["rv"]
  if fn is not None:
    return fn(*args, **kwargs)
  else:
    raise NotImplementedError


def factor(*args, **kwargs):
  fn = get_backend()["factor"]
  if fn is not None:
    return fn(*args, **kwargs)
  else:
    raise NotImplementedError


def prng_key():
  fn = get_backend()["prng_key"]
  if fn is not None:
    return fn()
  else:
    return None
