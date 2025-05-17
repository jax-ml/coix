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

"""Tests for oryx.py."""

import coix
import coix.core

try:
  import coix.oryx as coryx
except (ModuleNotFoundError, ImportError):
  coryx = None
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

pytest.skip("oryx backend is broken", allow_module_level=True)


def test_call_and_reap_tags():
  coix.set_backend("coix.oryx")

  def model(key):
    return coryx.rv(dist.Normal(0, 1), name="x")(key)

  _, trace, _ = coix.traced_evaluate(model)(random.PRNGKey(0))
  assert set(trace.keys()) == {"x"}
  assert set(trace["x"].keys()) == {"value", "log_prob"}


def test_delta_distribution():
  coix.set_backend("coix.oryx")

  def model(key):
    x = random.normal(key)
    return coryx.rv(dist.Delta(x, 5.0), name="x")(key)

  _, trace, _ = coix.traced_evaluate(model)(random.PRNGKey(0))
  assert set(trace.keys()) == {"x"}


def test_detach():
  coix.set_backend("coix.oryx")

  def model(x):
    return coryx.rv(dist.Delta(x, 0.0), name="x")(None) * x

  x = 2.0
  np.testing.assert_allclose(jax.grad(coix.detach(model))(x), x)


def test_detach_vmap():
  coix.set_backend("coix.oryx")

  def model(x):
    return coryx.rv(dist.Normal(x, 1.0), name="x")(random.PRNGKey(0))

  outs = coix.detach(jax.vmap(model))(jnp.ones(2))
  np.testing.assert_allclose(outs[0], outs[1])


def test_distribution():
  coix.set_backend("coix.oryx")

  def model(key):
    x = random.normal(key)
    return coryx.rv(dist.Delta(x, 5.0), name="x")(key)

  f = coix.oryx.call_and_reap_tags(
      coix.oryx.tag_distribution(model), coix.oryx.DISTRIBUTION
  )
  assert set(f(random.PRNGKey(0))[1][coix.oryx.DISTRIBUTION].keys()) == {"x"}


def test_empirical_program():
  coix.set_backend("coix.oryx")

  def model(x):
    trace = {
        "x": {"value": x, "log_prob": 11.0},
        "y": {"value": x + 1, "log_prob": 9.0, "is_observed": True},
    }
    return coix.empirical(0.0, trace, {})()

  _, trace, _ = coix.traced_evaluate(model)(1.0)
  samples = {name: site["value"] for name, site in trace.items()}
  jax.tree.map(np.testing.assert_allclose, samples, {"x": 1.0, "y": 2.0})
  assert "is_observed" not in trace["x"]
  assert trace["y"]["is_observed"]


def test_factor():
  coix.set_backend("coix.oryx")

  def model(x):
    return coryx.factor(x, name="x")

  _, trace, _ = coix.traced_evaluate(model)(10.0)
  assert "x" in trace
  np.testing.assert_allclose(trace["x"]["log_prob"], 10.0)


def test_log_prob_detach():
  coix.set_backend("coix.oryx")

  def model(loc):
    x = coryx.rv(dist.Normal(loc, 1), name="x")(random.PRNGKey(0))
    return x

  def actual_fn(x):
    return coix.traced_evaluate(coix.detach(model))(x)[1]["x"]["log_prob"]

  def expected_fn(x):
    return dist.Normal(x, 1).log_prob(model(1.0))

  actual = jax.grad(actual_fn)(1.0)
  expect = jax.grad(expected_fn)(1.0)
  np.testing.assert_allclose(actual, expect)


def test_observed():
  coix.set_backend("coix.oryx")

  def model(a):
    return coryx.rv(dist.Delta(a, 3.0), obs=1.0, name="x") + a

  _, trace, _ = coix.traced_evaluate(model)(2.0)
  assert "x" in trace
  np.testing.assert_allclose(trace["x"]["value"], 1.0)
  assert trace["x"]["is_observed"]


def test_stick_the_landing():
  coix.set_backend("coix.oryx")

  def model(lp):
    return coryx.rv(dist.Delta(0.0, lp), name="x")(None)

  def p(x):
    return coix.traced_evaluate(coix.detach(model))(x)[1]["x"]["log_prob"]

  def q(x):
    model_stl = coix.detach(coix.stick_the_landing(model))
    return coix.traced_evaluate(model_stl)(x)[1]["x"]["log_prob"]

  np.testing.assert_allclose(jax.grad(p)(5.0), 1.0)
  np.testing.assert_allclose(jax.grad(q)(5.0), 0.0)


def test_substitute():
  coix.set_backend("coix.oryx")

  def model(key):
    return coryx.rv(dist.Delta(1.0, 5.0), name="x")(key)

  expected = {"x": 9.0}
  _, trace, _ = coix.traced_evaluate(model, expected)(random.PRNGKey(0))
  actual = {"x": trace["x"]["value"]}
  jax.tree.map(np.testing.assert_allclose, actual, expected)


def test_suffix():
  coix.set_backend("coix.oryx")

  def model(x):
    return coryx.rv(dist.Delta(x, 5.0), name="x")(None)

  f = coix.oryx.call_and_reap_tags(
      coix.core.suffix(model), coix.oryx.RANDOM_VARIABLE
  )
  jax.tree.map(
      np.testing.assert_allclose,
      f(1.0)[1][coix.oryx.RANDOM_VARIABLE],
      {"x_PREV_": 1.0},
  )
