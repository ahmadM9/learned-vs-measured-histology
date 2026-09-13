import sys
from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).parents[1] / "data" / "kpmp"

# analysis scripts are plain files, not a package; tests import them
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))


def pytest_collection_modifyitems(config, items):
    if DATA_ROOT.exists() and any(DATA_ROOT.iterdir()):
        return
    skip = pytest.mark.skip(reason="KPMP sections not linked under data/kpmp")
    for item in items:
        if "requires_data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def data_root() -> Path:
    return DATA_ROOT
