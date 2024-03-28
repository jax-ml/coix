"""Tests for util.py."""

import coix
import jax
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, None])
def test_systematic_resampling_uniform(seed):
  log_weights = np.zeros(5)
  rng_key = jax.random.PRNGKey(seed) if seed is not None else None
  num_samples = 5
  resample_indices = coix.util.get_systematic_resampling_indices(
      log_weights, rng_key, num_samples
  )
  np.testing.assert_allclose(resample_indices, np.arange(5))
