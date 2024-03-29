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

"""Program transforms."""

import importlib

__all__ = [
    "detach",
    "empirical",
    "prng_key",
    "register_backend",
    "set_backend",
    "stick_the_landing",
    "suffix",
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
    prng_key=None,
    detach=None,
    stick_the_landing=None,
):
  """Register backend."""
  fn_map = {
      "traced_evaluate": traced_evaluate,
      "empirical": empirical,
      "suffix": suffix,
      "prng_key": prng_key,
      "detach": detach,
      "stick_the_landing": stick_the_landing,
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
        "prng_key",
        "detach",
        "stick_the_landing",
    ]:
      fn_map[fn] = getattr(module, fn, None)
    register_backend(backend, **fn_map)

  _COIX_BACKEND = backend


def get_backend_name():
  return _COIX_BACKEND


def get_backend():
  backend = _COIX_BACKEND
  if backend is None:
    set_backend("coix.numpyro")
    return _BACKENDS["coix.numpyro"]
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


def traced_evaluate(p, latents=None, seed=None, **kwargs):
  """Performs traced evaluation for a program `p`."""
  # Work around some backends not having `seed` keyword.
  kwargs = kwargs.copy()
  if seed is not None:
    kwargs["seed"] = seed
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


def prng_key():
  fn = get_backend()["prng_key"]
  if fn is not None:
    return fn()
  else:
    return None
