"""Configurations frozen after cross-validation (HYPOTHESIS.md, change 11). Used by the test evaluation.

Changing anything here changes FROZEN_HASH, and test_eval.py refuses to run against an existing
test lock with a different hash: the test set is used for ONE configuration only.
"""
from __future__ import annotations

import hashlib
import json

from src.models import GBMQuantile, HIBaseline, PopulationPrior

LEAD_TIME = 20                 # logistics lead time L (cycles), assumption, see HYPOTHESIS.md
LEAD_TIMES = (10, 20, 30)      # sensitivity analysis

FROZEN = {
    "B0": ("PopulationPrior", {}),
    "B1": ("HIBaseline", {"W": 30, "r_max": 100, "intervals": "conformal"}),
    "M1": ("GBMQuantile", {"cap": 125, "sensors": "all", "features": "full", "intervals": "cqr"}),
}
CV_KEYS = {"B0": "B0", "B1": "B1_W30_R100_conformal", "M1": "M1_cap125_allsensors_cqr"}

_CLASSES = {"PopulationPrior": PopulationPrior, "HIBaseline": HIBaseline, "GBMQuantile": GBMQuantile}


def make(name: str):
    cls, kw = FROZEN[name]
    return _CLASSES[cls](**kw)


FROZEN_HASH = hashlib.sha256(json.dumps({"models": FROZEN, "L": LEAD_TIME}, sort_keys=True).encode()).hexdigest()[:16]
