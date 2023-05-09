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
