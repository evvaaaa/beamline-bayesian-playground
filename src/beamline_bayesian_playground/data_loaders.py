from pathlib import Path

import pandas as pd


def load_spec_dat(path: Path) -> pd.DataFrame:
    path = Path(__file__).parent / "real_data" / path
    if not path.exists():
        raise ValueError(f"{path} does not exist.")
    return pd.read_csv(path, sep=r"\s+", engine="python")
