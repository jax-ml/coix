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
