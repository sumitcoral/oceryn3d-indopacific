import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
sys.path.insert(0, HERE)


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    """Synthetic raw data + the pipeline run through the cache (Stages 1-5)."""
    import synthetic
    import build_cache
    import build_forcing
    import build_index
    import build_static
    import compute_stats
    root = str(tmp_path_factory.mktemp("world"))
    synthetic.write_world(root)
    cfg = synthetic.write_config(root)
    assert build_forcing.main(["--config", cfg]) == 0
    assert build_index.main(["--config", cfg]) == 0
    assert build_static.main(["--config", cfg]) == 0
    assert compute_stats.main(["--config", cfg, "--stride", "1"]) == 0
    assert build_cache.main(["--config", cfg]) == 0
    return {"root": root, "cfg": cfg}
