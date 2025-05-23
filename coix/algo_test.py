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

"""Tests for algo.py."""

import functools

import coix
from jax import random
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import optax

coix.set_backend("coix.numpyro")

np.random.seed(0)
num_data, dim = 4, 2
data = np.random.randn(num_data, dim).astype(np.float32)
loc_p = np.random.randn(dim).astype(np.float32)
precision_p = np.random.rand(dim).astype(np.float32)
scale_p = np.sqrt(1 / precision_p)
precision_x = np.random.rand(dim).astype(np.float32)
scale_x = np.sqrt(1 / precision_x)
precision_q = precision_p + num_data * precision_x
loc_q = (data.sum(0) * precision_x + loc_p * precision_p) / precision_q
log_scale_q = -0.5 * np.log(precision_q)


def vmap(p):
  return numpyro.handlers.plate("N", 5)(p)


def model(params, key):
  del params
  key_z, key_next = random.split(key)
  z = numpyro.sample("z", dist.Normal(loc_p, scale_p).to_event(), rng_key=key_z)
  z = jnp.repeat(z[..., None, :], num_data, axis=-2)
  x = numpyro.sample("x", dist.Normal(z, scale_x).to_event(2), obs=data)
  return key_next, z, x


def guide(params, key, *args):
  del args
  key, _ = random.split(key)  # split here to test tie_in
  scale_q = jnp.exp(params["log_scale_q"])
  z = numpyro.sample(
      "z", dist.Normal(params["loc_q"], scale_q).to_event(), rng_key=key
  )
  return z


def check_ess(make_program):
  params = {"loc_q": loc_q, "log_scale_q": log_scale_q}
  p = vmap(functools.partial(model, params))
  q = vmap(functools.partial(guide, params))
  program = make_program(p, q)

  key = random.PRNGKey(0)
  ess = coix.traced_evaluate(program)(key)[2]["ess"]
  np.testing.assert_allclose(ess, 5.0)


def run_inference(make_program, num_steps=1000):
  """Performs inference given an algorithm `make_program`."""

  def loss_fn(params, key):
    p = vmap(functools.partial(model, params))
    q = vmap(functools.partial(guide, params))
    program = make_program(p, q)

    metrics = coix.traced_evaluate(program)(key)[2]
    return metrics["loss"], metrics

  init_params = {
      "loc_q": jnp.zeros_like(loc_q),
      "log_scale_q": jnp.zeros_like(log_scale_q),
  }
  params, _ = coix.util.train(
      loss_fn, init_params, optax.adam(0.01), num_steps=num_steps
  )

  np.testing.assert_allclose(params["loc_q"], loc_q, atol=0.2)
  np.testing.assert_allclose(params["log_scale_q"], log_scale_q, atol=0.2)


def test_apgs():
  check_ess(lambda p, q: coix.algo.apgs(p, [q]))
  run_inference(lambda p, q: coix.algo.apgs(p, [q]))


def test_rws():
  check_ess(coix.algo.rws)
  run_inference(coix.algo.rws)


def test_svi_elbo():
  check_ess(coix.algo.svi)
  run_inference(coix.algo.svi)


def test_svi_iwae():
  check_ess(coix.algo.svi_iwae)
  run_inference(coix.algo.svi_iwae)


def test_svi_stl():
  check_ess(coix.algo.svi_stl)
  run_inference(coix.algo.svi_stl)
