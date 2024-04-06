# coix

[![Unittests](https://github.com/jax-ml/coix/actions/workflows/pytest_and_autopublish.yml/badge.svg)](https://github.com/jax-ml/coix/actions/workflows/pytest_and_autopublish.yml)
[![Documentation Status](https://readthedocs.org/projects/coix/badge/?version=latest)](https://coix.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/coix.svg)](https://badge.fury.io/py/coix)

Coix (COmbinators In jaX) is a flexible and backend-agnostic implementation of inference combinators [(Stites and Zimmermann et al., 2021)](https://arxiv.org/abs/2103.00668), a set of program transformations for compositional inference with probabilistic programs. Coix ships with backends for numpyro and oryx, and a set of pre-implemented losses and utility functions that allows to implement and run a wide variety of inference algorithms out-of-the-box.

Coix is a lightweight framework which includes the following main components:

- **coix.api:** Implementation of the program combinators.
- **coix.core:** Basic program transformations which are used to modify behavior of a stochastic program.
- **coix.loss:** Common objectives for variational inference.
- **coix.algo:** Example inference algorithms.

Currently, we support [numpyro](https://github.com/pyro-ppl/numpyro) and [oryx](https://github.com/jax-ml/oryx) backends. But other backends can be easily added via the [coix.register_backend](https://coix.readthedocs.io/en/latest/core.html#coix.core.register_backend) utility.

*This is not an officially supported Google product.*

## Installation

To install Coix, you can use pip:

```
pip install coix
```

or you can clone the repository:

```
git clone https://github.com/jax-ml/coix.git
cd coix
pip install -e .[dev,doc]
```

Many examples would run faster on accelerators. You can follow the [JAX installation](https://jax.readthedocs.io/en/latest/installation.html) instruction for how to install JAX with GPU or TPU support.

