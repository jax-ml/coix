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

"""Inference objectives."""

from coix import util
import jax
import jax.numpy as jnp

__all__ = [
    "apg_loss",
    "avo_loss",
    "elbo_loss",
    "fkl_loss",
    "iwae_loss",
    "rkl_loss",
    "rws_loss",
]


def apg_loss(
    q_trace,
    p_trace,
    incoming_log_weight,
    incremental_log_weight,
    aggregate=True,
):
  """RWS objective that exploits conditional dependency."""
  del incoming_log_weight, incremental_log_weight
  p_log_probs = {
      name: util.get_site_log_prob(site) for name, site in p_trace.items()
  }
  q_log_probs = {
      name: util.get_site_log_prob(site) for name, site in q_trace.items()
  }
  forward_sites = [name[:-6] for name in p_trace if name.endswith("_PREV_")]
  observed = [
      name for name, site in p_trace.items() if util.is_observed_site(site)
  ]

  log_probs = [
      lp
      for name, lp in p_log_probs.items()
      if (name in forward_sites) or name in observed
  ]
  min_ndim = min(jnp.ndim(lp) for lp in log_probs)
  batch_shape = ()
  for i in range(min_ndim):
    dims = set(jnp.shape(lp)[i] for lp in log_probs)
    if len(dims) > 1:
      break
    batch_shape = batch_shape + tuple(dims)
  batch_ndims = len(batch_shape)

  global_sites = [
      name
      for name, lp in p_log_probs.items()
      if (not name.endswith("_PREV_"))
      and (lp.shape[:batch_ndims] != batch_shape)
  ]
  target_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in p_log_probs.items()
      if not (name.endswith("_PREV_") or name in global_sites)
  )
  reverse_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in p_log_probs.items()
      if name.endswith("_")
  )
  forward_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in q_log_probs.items()
      if name in forward_sites
  )
  proposal_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in q_log_probs.items()
      if (name not in forward_sites) and name not in global_sites
  )

  surrogate_loss = target_lp + forward_lp
  log_weight = target_lp + reverse_lp - (forward_lp + proposal_lp)
  w = jax.lax.stop_gradient(jax.nn.softmax(log_weight, axis=0))
  loss = -(w * surrogate_loss)
  if aggregate:
    loss = loss.sum()
  return loss


def avo_loss(
    q_trace,
    p_trace,
    incoming_log_weight,
    incremental_log_weight,
    aggregate=True,
):
  """Annealed Variational Objective."""
  del q_trace, p_trace
  surrogate_loss = incremental_log_weight
  if jnp.ndim(incoming_log_weight) > 0:
    w1 = 1.0 / incoming_log_weight.shape[0]
  else:
    w1 = 1.0
  loss = -(w1 * surrogate_loss)
  if aggregate:
    loss = loss.sum()
  return loss


def elbo_loss(
    q_trace,
    p_trace,
    incoming_log_weight,
    incremental_log_weight,
    aggregate=True,
):
  """Evidence Lower Bound objective."""
  del q_trace, p_trace
  surrogate_loss = incremental_log_weight
  if jnp.ndim(incoming_log_weight) > 0:
    w1 = jax.lax.stop_gradient(jax.nn.softmax(incoming_log_weight, axis=0))
  else:
    w1 = 1.0
  loss = -(w1 * surrogate_loss)
  if aggregate:
    loss = loss.sum()
  return loss


def fkl_loss(
    q_trace,
    p_trace,
    incoming_log_weight,
    incremental_log_weight,
    aggregate=True,
):
  """Forward KL objective. Here we do not optimize p."""
  del p_trace
  batch_ndims = incoming_log_weight.ndim
  q_log_probs = {
      name: util.get_site_log_prob(site) for name, site in q_trace.items()
  }
  proposal_sites = [
      name
      for name, site in q_trace.items()
      if name.endswith("_PREV_")
      or (isinstance(site, dict) and "suffix" in site)
  ]

  proposal_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in q_log_probs.items()
      if name in proposal_sites
  )
  forward_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in q_log_probs.items()
      if name not in proposal_sites
  )

  surrogate_loss = forward_lp + proposal_lp
  w1 = jax.lax.stop_gradient(jax.nn.softmax(incoming_log_weight, axis=0))
  log_weight = incoming_log_weight + incremental_log_weight
  w = jax.lax.stop_gradient(jax.nn.softmax(log_weight, axis=0))
  loss = -(w * surrogate_loss - w1 * proposal_lp)
  if aggregate:
    loss = loss.sum()
  return loss


def iwae_loss(
    q_trace,
    p_trace,
    incoming_log_weight,
    incremental_log_weight,
    aggregate=True,
):
  """Importance Weighted Autoencoder objective."""
  del q_trace, p_trace
  log_weight = incoming_log_weight + incremental_log_weight
  surrogate_loss = log_weight
  if jnp.ndim(incoming_log_weight) > 0:
    w = jax.lax.stop_gradient(jax.nn.softmax(log_weight, axis=0))
  else:
    w = 1.0
  loss = -(w * surrogate_loss)
  if aggregate:
    loss = loss.sum()
  return loss


def rkl_loss(
    q_trace,
    p_trace,
    incoming_log_weight,
    incremental_log_weight,
    aggregate=True,
):
  """Reverse KL objective."""
  batch_ndims = incoming_log_weight.ndim
  p_log_probs = {
      name: util.get_site_log_prob(site) for name, site in p_trace.items()
  }
  q_log_probs = {
      name: util.get_site_log_prob(site) for name, site in q_trace.items()
  }
  proposal_sites = [
      name
      for name, site in q_trace.items()
      if name.endswith("_PREV_")
      or (isinstance(site, dict) and "suffix" in site)
  ]

  proposal_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in q_log_probs.items()
      if name in proposal_sites
  )
  target_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in p_log_probs.items()
      if not name.endswith("_PREV_")
  )

  w1 = jax.lax.stop_gradient(jax.nn.softmax(incoming_log_weight, axis=0))
  v = jax.lax.stop_gradient(incremental_log_weight)
  surrogate_loss = (
      incremental_log_weight + (1 + v - (w1 * v).sum(0)) * proposal_lp
  )
  log_weight = incoming_log_weight + incremental_log_weight
  w = jax.lax.stop_gradient(jax.nn.softmax(log_weight, axis=0))
  loss = -(w1 * surrogate_loss - w * target_lp)
  if aggregate:
    loss = loss.sum()
  return loss


def rws_loss(
    q_trace,
    p_trace,
    incoming_log_weight,
    incremental_log_weight,
    aggregate=True,
):
  """Reweighted Wake-Sleep objective."""
  batch_ndims = incoming_log_weight.ndim
  p_log_probs = {
      name: util.get_site_log_prob(site) for name, site in p_trace.items()
  }
  q_log_probs = {
      name: util.get_site_log_prob(site) for name, site in q_trace.items()
  }
  proposal_sites = [
      name
      for name, site in q_trace.items()
      if name.endswith("_PREV_")
      or (isinstance(site, dict) and "suffix" in site)
  ]

  proposal_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in q_log_probs.items()
      if name in proposal_sites
  )
  forward_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in q_log_probs.items()
      if name not in proposal_sites
  )
  target_lp = sum(
      lp.reshape(lp.shape[:batch_ndims] + (-1,)).sum(-1)
      for name, lp in p_log_probs.items()
      if not name.endswith("_PREV_")
  )

  surrogate_loss = (target_lp - proposal_lp) + forward_lp
  log_weight = incoming_log_weight + incremental_log_weight
  w = jax.lax.stop_gradient(jax.nn.softmax(log_weight, axis=0))
  loss = -(w * surrogate_loss)
  if aggregate:
    loss = loss.sum()
  return loss
