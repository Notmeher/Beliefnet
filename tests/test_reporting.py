import json
from pathlib import Path

from beliefnet.eval.reporting import build_details_md, save_result


def test_save_result_writes_json(tmp_path):
    path = save_result(tmp_path, "exp_demo", {
        "title": "Demo Experiment",
        "summary": {"acc": 1.0},
        "tables": {"t": [{"a": 1, "b": 2.0}]},
        "notes": ["a note"],
    })
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["experiment"] == "exp_demo"
    assert "generated" in data


def test_build_details_md_aggregates(tmp_path):
    save_result(tmp_path, "exp_a", {"title": "Alpha", "summary": {"x": 0.5}})
    save_result(tmp_path, "exp_b", {"title": "Beta",
                                    "tables": {"tab": [{"k": 1}]}})
    md = build_details_md(tmp_path)
    text = Path(md).read_text(encoding="utf-8")
    assert "Alpha" in text and "Beta" in text
    assert "## Contents" in text
