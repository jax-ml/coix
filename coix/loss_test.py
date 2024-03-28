"""Tests for loss.py."""

import coix
import jax.numpy as jnp
import numpy as np

p_trace = {
    "x": {"log_prob": np.full((3, 2), 2.0)},
    "y": {"log_prob": np.array([3.0, 0.0, -2.0])},
    "x_PREV_": {"log_prob": np.ones((3, 2))},
}
q_trace = {
    "x": {"log_prob": np.ones((3, 2))},
    "y": {"log_prob": np.array([1.0, 1.0, 0.0])},
    "x_PREV_": {"log_prob": np.full((3, 2), 3.0)},
}
incoming_weight = np.zeros(3)
incremental_weight = np.log(np.array([1 / 6, 1 / 3, 1 / 2]))


def test_apg():
  result = coix.loss.apg_loss(
      q_trace, p_trace, incoming_weight, incremental_weight
  )
  np.testing.assert_allclose(result, -6.0)


def test_elbo():
  result = coix.loss.elbo_loss(
      q_trace, p_trace, incoming_weight, incremental_weight
  )
  expected = -incremental_weight.sum() / 3
  np.testing.assert_allclose(result, expected)


def test_iwae():
  result = coix.loss.iwae_loss(
      q_trace, p_trace, incoming_weight, incremental_weight
  )
  w = incoming_weight + incremental_weight
  expected = -(jnp.exp(w) * w).sum()
  np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_rws():
  result = coix.loss.rws_loss(
      q_trace, p_trace, incoming_weight, incremental_weight
  )
  np.testing.assert_allclose(result, 1.0, rtol=1e-6)
