# Copyright 2023 The coix Authors.
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

"""Test for util.py."""

from coix import util
import jax
import numpy as np
import pytest


@pytest.mark.parametrize("seed", [0, None])
def test_systematic_resampling_uniform(seed):
  log_weights = np.zeros(5)
  rng_key = jax.random.PRNGKey(seed) if seed is not None else None
  num_samples = 5
  resample_indices = util.get_systematic_resampling_indices(
      log_weights, rng_key, num_samples)
  np.testing.assert_allclose(resample_indices, np.arange(5))
