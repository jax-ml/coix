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

"""Inference algorithms."""

import functools

from coix.api import compose
from coix.api import extend
from coix.api import fori_loop
from coix.api import propose
from coix.api import resample
from coix.core import detach
from coix.core import stick_the_landing
from coix.loss import apg_loss
from coix.loss import avo_loss
from coix.loss import elbo_loss
from coix.loss import fkl_loss
from coix.loss import iwae_loss
from coix.loss import rkl_loss
from coix.loss import rws_loss


__all__ = [
    "aft",
    "apgs",
    "dais",
    "nasmc",
    "nvi_avo",
    "nvi_fkl",
    "nvi_rkl",
    "rws",
    "svi",
    "svi_iwae",
    "svi_stl",
    "vsmc",
]


def _use_fori_loop(targets, num_targets, *fns):
  if callable(targets):
    if num_targets is None:
      raise ValueError("To use fori_loop, num_targets needs to be specified.")
    for fn in fns:
      if not callable(fn):
        raise ValueError(
            "To use fori_loop, input programs need to be callable,"
            f" but got {type(fn)}."
        )
    return True
  return False


def aft(targets, flows, *, num_targets=None):
  """Annealed Flow Transport.

  [1] Annealed Flow Transport Monte Carlo,
      Michael Arbel, Alexander G. D. G. Matthews, Arnaud Doucet
      https://arxiv.org/abs/2102.07501

  Args:
    targets: a list of target programs
    flows: a list of flows
    num_targets: the number of targets

  Returns:
    q: the inference program
  """
  if _use_fori_loop(targets, num_targets, flows):

    def body_fun(i, q):
      p, q = targets(i + 1), compose(flows(i), resample(q))
      return propose(p, q, loss_fn=elbo_loss, detach=True)

    return fori_loop(0, num_targets - 1, body_fun, targets(0))

  q = targets[0]
  for p, flow in zip(targets[1:], flows):
    q = propose(p, compose(flow, resample(q)), loss_fn=elbo_loss, detach=True)
  return q


def apgs(target, kernels, *, num_sweeps=1):
  """Amortized Population Gibbs Sampler.

  [1] Amortized Population Gibbs Samplers with Neural Sufficient Statistics,
      Hao Wu, Heiko Zimmermann, Eli Sennesh, Tuan Anh Le, Jan-Willem van de
      Meent
      https://arxiv.org/abs/1911.01382

  Args:
    target: the target program
    kernels: the Gibbs kernels
    num_sweeps: the number of sweeps

  Returns:
    q: the inference program
  """
  kernels = [detach(k) for k in kernels]
  q = functools.reduce(lambda a, b: compose(b, a), kernels[1:], kernels[0])
  q = propose(target, q, loss_fn=rws_loss)

  def body_fn(_, q):
    for k in kernels:
      q = compose(k, resample(q))
      q = propose(extend(target, k), q, loss_fn=apg_loss)
    return q

  return fori_loop(0, num_sweeps, body_fn, q)


def dais(targets, momentum, leapfrog, refreshment, *, num_targets=None):
  """Differentiable Annealed Importance Sampling.

  [1] MCMC Variational Inference via Uncorrected Hamiltonian Annealing,
      Tomas Geffner, Justin Domke
      https://arxiv.org/abs/2107.04150
  [2] Differentiable Annealed Importance Sampling and the Perils of Gradient
      Noise,
      Guodong Zhang, Kyle Hsu, Jianing Li, Chelsea Finn, Roger Grosse
      https://arxiv.org/abs/2107.10211

  Args:
    targets: a list of target programs
    momentum: the momentum program which calculates kinetic energy
    leapfrog: the program which performs leapfrog update
    refreshment: the momentum refreshment program
    num_targets: the number of targets

  Returns:
    q: the inference program
  """
  if _use_fori_loop(targets, num_targets):

    def body_fun(i, q):
      p = extend(compose(momentum, targets(i), suffix=False), refreshment)
      return propose(p, compose(refreshment, compose(leapfrog, q)))

    q = compose(momentum, targets(0), suffix=False)
    q = fori_loop(1, num_targets - 1, body_fun, q)
    p = compose(momentum, targets(num_targets - 1), suffix=False)
    q = compose(refreshment, compose(leapfrog, q))
    return propose(extend(p, refreshment), q, loss_fn=iwae_loss)

  targets = [compose(momentum, p, suffix=False) for p in targets]
  q = targets[0]
  loss_fns = [None] * (len(targets) - 2) + [iwae_loss]
  for p, loss_fn in zip(targets[1:], loss_fns):
    q = compose(refreshment, compose(leapfrog, q))
    q = propose(extend(p, refreshment), q, loss_fn=loss_fn)
  return q


def nasmc(targets, proposals, *, num_targets=None):
  """Neural Adaptive Sequential Monte Carlo.

  [1] Neural Adaptive Sequential Monte Carlo,
      Shixiang Gu, Zoubin Ghahramani, Richard E. Turner
      https://arxiv.org/abs/1506.03338

  Args:
    targets: a list of target programs
    proposals: a list of proposal programs
    num_targets: the number of targets

  Returns:
    q: the inference program
  """
  if _use_fori_loop(targets, num_targets, proposals):

    def body_fun(i, q):
      p, q = targets(i), compose(detach(proposals(i)), resample(q))
      return propose(p, q, loss_fn=rws_loss)

    q = propose(targets(0), detach(proposals(0)), loss_fn=rws_loss)
    return fori_loop(1, num_targets, body_fun, q)

  q = propose(targets[0], detach(proposals[0]), loss_fn=rws_loss)
  for p, fwd in zip(targets[1:], proposals[1:]):
    q = propose(p, compose(detach(fwd), resample(q)), loss_fn=rws_loss)
  return q


def nvi_avo(targets, forwards, reverses, *, num_targets=None):
  """AIS with Annealed Variational Objective.

  [1] Improving Explorability in Variational Inference with Annealed Variational
      Objectives,
      Chin-Wei Huang, Shawn Tan, Alexandre Lacoste, Aaron Courville
      https://arxiv.org/abs/1809.01818

  Args:
    targets: a list of target programs
    forwards: a list of forward kernels
    reverses: a list of reverse kernels
    num_targets: the number of targets

  Returns:
    q: the inference program
  """
  if _use_fori_loop(targets, num_targets, forwards, reverses):

    def body_fun(i, q):
      p, q = extend(targets(i + 1), reverses(i)), compose(forwards(i), q)
      return propose(p, q, loss_fn=avo_loss, detach=True)

    return fori_loop(0, num_targets - 1, body_fun, targets(0))

  q = targets[0]
  for p, fwd, rev in zip(targets[1:], forwards, reverses):
    q = propose(extend(p, rev), compose(fwd, q), loss_fn=avo_loss, detach=True)
  return q


def nvi_fkl(targets, proposals, *, num_targets=None):
  """Nested Variational Inference with forward KL objective.

  Note: The implementation assumes that targets are smoothing distributions.
  This is different from `nasmc`, where we assume that the targets are filtering
  distributions. We also assume that the final target does not have parameters.

  [1] Nested Variational Inference,
      Heiko Zimmermann, Hao Wu, Babak Esmaeili, Jan-Willem van de Meent
      https://arxiv.org/abs/2106.11302

  Args:
    targets: a list of target programs
    proposals: the proposal for the initial target
    num_targets: the number of targets

  Returns:
    q: the inference program
  """
  if _use_fori_loop(targets, num_targets, proposals):

    def body_fun(i, q):
      p, q = targets(i), compose(detach(proposals(i)), resample(q))
      return propose(p, q, loss_fn=fkl_loss)

    q = propose(targets(0), detach(proposals(0)), loss_fn=fkl_loss)
    return fori_loop(1, num_targets, body_fun, q)

  q = propose(targets[0], detach(proposals[0]), loss_fn=fkl_loss)
  for p, fwd in zip(targets[1:], proposals[1:]):
    q = propose(p, compose(detach(fwd), resample(q)), loss_fn=fkl_loss)
  return q


def nvi_rkl(targets, forwards, reverses, *, num_targets=None):
  """Nested Variational Inference with reverse KL objective.

  If `targets` is a callable which takes an integer index and returns the i-th
  taget, we will use `fori_loop` combinator to improve the compiling time. This
  requires `num_targets` to be a concrete value.

  Note: In nested VI, we typically assume that the final target does not have
  parameters. This allows us to optimize intermediate KLs to bridge from the
  initial target to the final target. Here we use ELBO loss in the last step
  to also maximize likelihood in case there are parameters in the final target.

  [1] Nested Variational Inference,
      Heiko Zimmermann, Hao Wu, Babak Esmaeili, Jan-Willem van de Meent
      https://arxiv.org/abs/2106.11302

  Args:
    targets: a list of target programs
    forwards: a list of forward kernels
    reverses: a list of reverse kernels
    num_targets: the number of targets

  Returns:
    q: the inference program
  """
  if _use_fori_loop(targets, num_targets, forwards, reverses):

    def body_fun(i, q):
      p, fwd, rev = targets(i + 1), forwards(i), reverses(i)
      p, q = extend(p, rev), compose(stick_the_landing(fwd), resample(q))
      return propose(p, q, loss_fn=rkl_loss, detach=True)

    return fori_loop(0, num_targets - 1, body_fun, targets(0))

  q = targets[0]
  for p, fwd, rev in zip(targets[1:], forwards, reverses):
    p, q = extend(p, rev), compose(stick_the_landing(fwd), resample(q))
    q = propose(p, q, loss_fn=rkl_loss, detach=True)
  return q


def rws(target, proposal):
  """Reweighted Wake-Sleep.

  [1] Reweighted Wake-Sleep,
      Jörg Bornschein, Yoshua Bengio
      https://arxiv.org/abs/1406.2751
  [2] Revisiting Reweighted Wake-Sleep for Models with Stochastic Control Flow,
      Tuan Anh Le, Adam R. Kosiorek, N. Siddharth, Yee Whye Teh, Frank Wood
      https://arxiv.org/abs/1805.10469

  Args:
    target: the target program
    proposal: the proposal program

  Returns:
    q: the inference program
  """
  return propose(target, detach(proposal), loss_fn=rws_loss)


def svi(target, proposal):
  """Stochastic Variational Inference.

  [1] Auto-Encoding Variational Bayes,
      Diederik P Kingma, Max Welling
      https://arxiv.org/abs/1312.6114
  [2] Stochastic Backpropagation and Approximate Inference in Deep Generative
      Models,
      Danilo Jimenez Rezende, Shakir Mohamed, Daan Wierstra
      https://arxiv.org/abs/1401.4082

  Args:
    target: the target program
    proposal: the proposal program

  Returns:
    q: the inference program
  """
  return propose(target, proposal, loss_fn=elbo_loss)


def svi_iwae(target, proposal):
  """SVI with Important Weighted Autoencoder objective.

  [1] Importance Weighted Autoencoders,
      Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
      https://arxiv.org/abs/1509.00519

  Args:
    target: the target program
    proposal: the proposal program

  Returns:
    q: the inference program
  """
  return propose(target, proposal, loss_fn=iwae_loss)


def svi_stl(target, proposal):
  """SVI with Sticking-the-Landing objective.

  [1] Sticking the Landing: Simple, Lower-Variance Gradient Estimators for
      Variational Inference,
      Geoffrey Roeder, Yuhuai Wu, David Duvenaud
      https://arxiv.org/abs/1703.09194

  Args:
    target: the target program
    proposal: the proposal program

  Returns:
    q: the inference program
  """
  return propose(target, stick_the_landing(proposal), loss_fn=elbo_loss)


def vsmc(targets, proposals, *, num_targets=None):
  """Variational Sequential Monte Carlo.

  Note: Here, we assume that the dimension of variables is constant (modulo
  masking) during SMC steps. The targets can be filtering distributions or
  smoothing distributions (as in [4]).

  [1] Filtering Variational Objectives,
      Chris J. Maddison, Dieterich Lawson, George Tucker, Nicolas Heess,
      Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, Yee Whye Teh
      https://arxiv.org/abs/1705.09279
  [2] Auto-Encoding Sequential Monte Carlo,
      Tuan Anh Le, Maximilian Igl, Tom Rainforth, Tom Jin, Frank Wood
      https://arxiv.org/abs/1705.10306
  [3] Variational Sequential Monte Carlo,
      Christian A. Naesseth, Scott W. Linderman, Rajesh Ranganath, David M. Blei
      https://arxiv.org/abs/1705.11140
  [4] Twisted Variational Sequential Monte Carlo,
      Dieterich Lawson, George Tucker, Christian A Naesseth, Chris J Maddison,
      Ryan P Adams, Yee Whye Teh
      http://bayesiandeeplearning.org/2018/papers/111.pdf

  Args:
    targets: a list of target programs
    proposals: a list of proposal programs
    num_targets: the number of targets

  Returns:
    q: the inference program
  """
  if _use_fori_loop(targets, num_targets, proposals):

    def body_fun(i, q):
      return compose(proposals(i + 1), resample(propose(targets(i), q)))

    q = fori_loop(0, num_targets - 1, body_fun, proposals(0))
    return propose(targets(num_targets - 1), q, loss_fn=iwae_loss)

  q = propose(targets[0], proposals[0])
  loss_fns = [None] * (len(proposals) - 2) + [iwae_loss]
  for p, fwd, loss_fn in zip(targets[1:], proposals[1:], loss_fns):
    q = propose(p, compose(fwd, resample(q)), loss_fn=loss_fn)
  return q
