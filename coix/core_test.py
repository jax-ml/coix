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

"""Tests for core.py."""

import coix.core


def test_desuffix():
  trace = {
      "z_PREV__PREV_": 0,
      "v_PREV__PREV_": 1,
      "z_PREV_": 2,
      "v_PREV_": 3,
      "v": 4,
  }
  desuffix_trace = {
      "z_PREV_": 0,
      "v_PREV__PREV_": 1,
      "z": 2,
      "v_PREV_": 3,
      "v": 4,
  }
  assert coix.core.desuffix(trace) == desuffix_trace
