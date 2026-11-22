"""Loading of the raw NASA C-MAPSS files (read-only, never modified)."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "CMAPSSData"

SETTINGS = ["os1", "os2", "os3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle"] + SETTINGS + SENSORS  # 26 columns


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The NASA C-MAPSS files are not stored in the repository: download them "
            "from https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data and put "
            "train_FD001.txt, test_FD001.txt and RUL_FD001.txt in CMAPSSData/ (see README).")
    return path


def _read(path: Path) -> pd.DataFrame:
    _require(path)
    # Files are space separated with trailing spaces -> regex separator.
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name}: expected {len(COLUMNS)} columns, got {df.shape[1]}")
    df.columns = COLUMNS
    return df


def load_train(subset: str = "FD001") -> pd.DataFrame:
    return _read(DATA_DIR / f"train_{subset}.txt")


def load_test(subset: str = "FD001") -> pd.DataFrame:
    return _read(DATA_DIR / f"test_{subset}.txt")


def load_test_rul(subset: str = "FD001") -> pd.Series:
    rul = pd.read_csv(_require(DATA_DIR / f"RUL_{subset}.txt"), header=None, sep=r"\s+").iloc[:, 0]
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    return rul.rename("RUL")


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Target for run-to-failure trajectories: cycles remaining until the last observed cycle."""
    out = df.copy()
    out["RUL"] = out.groupby("unit")["cycle"].transform("max") - out["cycle"]
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(_require(path).read_bytes()).hexdigest()
