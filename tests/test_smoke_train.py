import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("tokenizers")

from mtt.eval import main as eval_main  # noqa: E402
from mtt.train import main as train_main  # noqa: E402


def test_train_and_eval_smoke(tmp_path):
    runs = tmp_path / "runs"
    for mode in ("wlt_only", "mtt"):
        train_main(["--mode", mode, "--smoke", "--out-dir", str(runs), "--device", "cpu"])
        assert (runs / mode / "model.pt").exists()
    eval_main(["--runs", "wlt_only", "mtt", "--runs-dir", str(runs), "--out-dir", str(tmp_path / "res"),
               "--device", "cpu"])
    res = json.loads((tmp_path / "res" / "results.json").read_text())
    assert set(res) == {"wlt_only", "mtt"}
    assert (tmp_path / "res" / "accuracy.png").exists()
