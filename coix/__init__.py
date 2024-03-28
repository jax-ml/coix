"""coix API."""

from coix import algo
from coix import loss
from coix import util
from coix.api import compose
from coix.api import extend
from coix.api import fori_loop
from coix.api import memoize
from coix.api import propose
from coix.api import resample
from coix.core import detach
from coix.core import empirical
from coix.core import factor
from coix.core import register_backend
from coix.core import rv
from coix.core import set_backend
from coix.core import stick_the_landing
from coix.core import traced_evaluate

__all__ = [
    "__version__",
    "empirical",
    "algo",
    "compose",
    "detach",
    "extend",
    "factor",
    "fori_loop",
    "loss",
    "memoize",
    "propose",
    "register_backend",
    "resample",
    "rv",
    "set_backend",
    "stick_the_landing",
    "traced_evaluate",
    "util",
]

# A new PyPI release will be pushed everytime `__version__` is increased
# When changing this, also update the CHANGELOG.md
__version__ = "0.0.1"
