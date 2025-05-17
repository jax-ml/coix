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

"""Tests for api.py."""

import coix
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

coix.set_backend("coix.numpyro")


def test_compose():
  def p(key):
    key, subkey = random.split(key)
    x = numpyro.sample("x", dist.Normal(0, 1), rng_key=subkey)
    return key, x

  def f(key, x):
    return numpyro.sample("z", dist.Normal(x, 1), rng_key=key)

  _, p_trace, _ = coix.traced_evaluate(coix.compose(f, p))(random.PRNGKey(0))
  assert set(p_trace.keys()) == {"x", "z"}


def test_extend():
  def p(key):
    key, subkey = random.split(key)
    x = numpyro.sample("x", dist.Normal(0, 1), rng_key=subkey)
    return key, x

  def f(key, x):
    return (numpyro.sample("z", dist.Normal(x, 1), rng_key=key),)

  def g(z):
    return z + 1

  key = random.PRNGKey(0)
  out, trace, _ = coix.traced_evaluate(coix.extend(p, f))(key)
  assert set(trace.keys()) == {"x", "z"}

  expected_key, expected_x = p(key)
  expected_key = random.key_data(expected_key)
  actual_key = random.key_data(out[0])
  np.testing.assert_allclose(actual_key, expected_key)
  np.testing.assert_allclose(out[1], expected_x)

  marginal_pfg = coix.traced_evaluate(coix.extend(p, coix.compose(g, f)))(key)[
      0
  ]
  actual_key2, actual_x2 = marginal_pfg
  actual_key2 = random.key_data(actual_key2)
  np.testing.assert_allclose(actual_key2, expected_key)
  np.testing.assert_allclose(actual_x2, expected_x)


def test_propose():
  def p(key):
    key, subkey = random.split(key)
    x = numpyro.sample("x", dist.Normal(0, 1), rng_key=subkey)
    return key, x

  def f(key, x):
    return numpyro.sample("z", dist.Normal(x, 1), rng_key=key)

  def q(key):
    return numpyro.sample("x", dist.Normal(1, 2), rng_key=key)

  program = coix.propose(coix.extend(p, f), q)
  key = random.PRNGKey(0)
  out, trace, metrics = coix.traced_evaluate(program)(key)
  assert set(trace.keys()) == {"x", "z"}
  assert isinstance(out, tuple) and len(out) == 2
  assert out[0].shape == key.shape
  with np.testing.assert_raises(AssertionError):
    np.testing.assert_allclose(metrics["log_density"], 0.0)

  def vmap(p):
    return numpyro.handlers.plate("N", 3)(p)

  particle_program = coix.propose(vmap(coix.extend(p, f)), vmap(q))
  particle_out = particle_program(key)
  assert isinstance(particle_out, tuple) and len(particle_out) == 2
  assert particle_out[1].shape == (3,)


def test_resample():
  def q(key):
    return numpyro.sample("x", dist.Normal(1, 2), rng_key=key)

  particle_program = numpyro.handlers.plate("N", 3)(q)
  key = random.PRNGKey(0)
  particle_out = coix.resample(particle_program)(key)
  assert particle_out.shape == (3,)


def test_resample_one():
  def q(key):
    x = numpyro.sample("x", dist.Normal(1, 2), rng_key=key)
    return numpyro.sample("z", dist.Normal(x, 1), obs=0.0)

  particle_program = numpyro.handlers.plate("N", 3)(q)
  key = random.PRNGKey(0)
  particle_out = coix.resample(particle_program, num_samples=())(key)
  assert not jnp.shape(particle_out)


def test_fori_loop():
  def drift(key, x):
    key_out, key = random.split(key)
    x_new = numpyro.sample("x", dist.Normal(x, 1.0), rng_key=key)
    return key_out, x_new

  compile_time = {"value": 0}

  def body_fun(_, q):
    compile_time["value"] += 1
    return coix.propose(drift, coix.compose(drift, q))

  q = drift
  for i in range(5):
    q = body_fun(i, q)
  x_init = np.zeros(3, np.float32)
  q(random.PRNGKey(0), x_init)
  assert compile_time["value"] == 5

  random_walk = coix.fori_loop(0, 5, body_fun, drift)
  random_walk(random.PRNGKey(0), x_init)
  assert compile_time["value"] == 6


# TODO(phandu): Support memoised arrays.
@pytest.mark.skip(reason="Currently, we only support memoised lists.")
def test_memoize():
  def model(key):
    x = numpyro.sample("x", dist.Normal(0, 1), rng_key=key)
    y = numpyro.sample("y", dist.Normal(x, 1), obs=0.0)
    return x, y

  def guide(key):
    return numpyro.sample("x", dist.Normal(1, 2), rng_key=key)

  def vmodel(key):
    return jax.vmap(model)(random.split(key, 5))

  def vguide(key):
    return jax.vmap(guide)(random.split(key, 3))

  memory = {"x": np.array([2, 4])}
  program = coix.memoize(vmodel, vguide, memory)
  out, trace, metrics = coix.traced_evaluate(program)(random.PRNGKey(0))
  assert set(trace.keys()) == {"x"}
  assert "memory" in metrics
  assert metrics["memory"]["x"].shape == (2,)
  assert out[0].shape == (2,)
