"""Stage-0 smoke tests. No torch / no HF / no data. Just package wiring."""
from __future__ import annotations


def test_import_package():
    import dpocurv

    assert hasattr(dpocurv, "__version__")
    assert isinstance(dpocurv.__version__, str)


def test_import_submodules():
    from dpocurv import types  # noqa: F401
    from dpocurv.utils import checkpoint, io, logging as ulog, seed  # noqa: F401
    from dpocurv import data, eval as deval, models, training, cli  # noqa: F401


def test_set_seed_idempotent():
    from dpocurv.utils.seed import set_seed

    set_seed(0)
    set_seed(0)


def test_derive_seed_stable():
    from dpocurv.utils.seed import derive_seed

    a = derive_seed(20260425, "split", "probe")
    b = derive_seed(20260425, "split", "probe")
    c = derive_seed(20260425, "split", "other")
    assert a == b
    assert a != c


def test_train_config_defaults():
    from dpocurv.types import TrainConfig

    cfg = TrainConfig(run_name="t", out_dir="/tmp/t")
    assert cfg.beta == 0.1
    assert cfg.curv_lambda == 0.0
    assert cfg.effective_batch == 16


def test_yaml_io_roundtrip(tmp_path):
    from dpocurv.utils.io import deep_merge, dump_yaml, load_yaml

    p = tmp_path / "x.yaml"
    payload = {"a": 1, "b": [1, 2], "nested": {"x": 1, "y": [3]}}
    dump_yaml(payload, p)
    assert load_yaml(p) == payload

    merged = deep_merge({"a": 1, "n": {"x": 1}}, {"a": 2, "n": {"y": 2}})
    assert merged == {"a": 2, "n": {"x": 1, "y": 2}}


def test_jsonl_writer(tmp_path):
    from dpocurv.utils.logging import JsonlMetricWriter

    p = tmp_path / "m.jsonl"
    with JsonlMetricWriter(p) as w:
        w.write({"step": 1, "loss": 0.5})
        w.write({"step": 2, "loss": 0.4})
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    import json

    rec0 = json.loads(lines[0])
    assert rec0["step"] == 1


def test_logger_idempotent():
    from dpocurv.utils.logging import get_logger

    a = get_logger("dpocurv.smoke")
    b = get_logger("dpocurv.smoke")
    assert a is b
