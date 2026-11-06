"""The app's plumbing - library, registry, series creation, jobs, API - with no model loaded."""
import sys
import time
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from manga import library, registry, series
from manga.jobs import Busy, JobRunner
from manga.service import with_trigger

REPO = Path(__file__).resolve().parents[1]


def _img(path, colour=(255, 255, 255), size=(64, 96)):
    Image.new("RGB", size, colour).save(path)
    return path


@pytest.fixture
def root(tmp_path):
    """A throwaway repo root: real configs, fake LoRA files."""
    (tmp_path / "configs").mkdir()
    for f in ("base.yaml", "dataset.yaml", "bungo.yaml", "bungo_eval.yaml"):
        (tmp_path / "configs" / f).write_text((REPO / "configs" / f).read_text("utf-8"), "utf-8")
    d = tmp_path / "models" / "loras" / "bungo"
    d.mkdir(parents=True)
    for s in ("step00001200", "step00002400"):
        (d / f"bungo_style-{s}.safetensors").write_bytes(b"x")
    return tmp_path


# ----------------------------------------------------------------- library

def test_library_dedupes_and_caps_least_recently_used(root, tmp_path):
    paths = []
    for i in range(3):
        paths.append(library.add(_img(tmp_path / f"{i}.png", (i * 40, 0, 0)), "refs", root,
                                 max_items=3))
        time.sleep(0.02)
    assert library.add(tmp_path / "0.png", "refs", root, max_items=3) == paths[0]   # same pixels
    time.sleep(0.02)
    library.touch(paths[1])                          # use order now: 1, 0, 2 -> 2 is LRU
    time.sleep(0.02)
    library.add(_img(tmp_path / "new.png", (0, 0, 255)), "refs", root, max_items=3)
    left = library.items("refs", root)
    assert len(left) == 3 and paths[2] not in left and paths[1] in left


def test_library_flattens_transparency(root, tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(p)
    out = Image.open(library.add(p, "outlines", root))
    assert out.mode == "RGB" and out.getpixel((5, 5)) == (255, 255, 255)


# ---------------------------------------------------------------- registry

def test_series_found_and_eval_configs_ignored(root):
    s = registry.series(root)
    assert set(s) == {"bungo"} and s["bungo"]["trigger"] == "bsdstyle"


def test_promote_round_trip_and_path_guard(root):
    ck = registry.checkpoints("bungo", root)
    assert len(ck) == 2 and all(c.startswith("models/loras/bungo/") for c in ck)
    registry.promote("bungo", ck[0], 0.8, root)
    assert registry.promoted("bungo", root)["lora"] == ck[0]
    with pytest.raises(ValueError):
        registry.safe_lora_path("../../etc/passwd.safetensors", root)


# ------------------------------------------------------------------ series

def test_create_series_from_pdf_and_zip(root, tmp_path):
    pdf = tmp_path / "Vol 1.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    z = tmp_path / "vol2.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("c001/001.jpg", b"jpg")
    cfg_path = series.create("vagabond", "vgbstyle", [pdf, z], head_drop=4, root=root)

    from manga.config import load
    cfg = load(str(cfg_path))
    assert cfg.series.trigger == "vgbstyle" and cfg.dataset.pages["head_drop"] == 4
    assert [v.get("pdf") or v.get("images") for v in cfg.series.volumes] == \
        ["data/volumes/vagabond/v1.pdf", "data/volumes/vagabond/v2"]
    assert (root / "data/volumes/vagabond/v2/c001/001.jpg").exists()
    assert cfg.lora.rank == 32 and cfg.dataset.filters["min_side"] == 448   # inherits dataset.yaml
    assert "vagabond" in registry.series(root)


def test_create_series_rejects_bad_input(root, tmp_path):
    with pytest.raises(ValueError):
        series.create("Bad Name", "", [tmp_path / "x.pdf"], root=root)
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as f:
        f.writestr("../../escape.jpg", b"x")
    with pytest.raises(ValueError):
        series.create("evil", "", [evil], root=root)


def test_suggested_steps_matches_bungo():
    assert series.suggested_steps(4697) == (11700, 1170)
    assert series.suggested_steps(500)[0] == 3000


def test_trigger_added_once():
    assert with_trigger("1boy, solo", "bsdstyle") == "bsdstyle, 1boy, solo"
    assert with_trigger("bsdstyle, 1boy", "bsdstyle") == "bsdstyle, 1boy"


# -------------------------------------------------------------------- jobs

def test_job_runs_logs_and_blocks_a_second_one(root):
    jobs = JobRunner(root)
    freed = []
    jobs.before_start = lambda: freed.append(1)
    jobs.start("sleep", [sys.executable, "-c", "import time; print('hi'); time.sleep(5)"])
    with pytest.raises(Busy):
        jobs.start("other", [sys.executable, "-c", "pass"])
    jobs.stop()
    assert not jobs.running() and freed == [1]
    assert "hi" in jobs.tail() or jobs.log.exists()


# --------------------------------------------------------------------- api

class FakeService:
    """Stands in for the GPU: returns blank images, records what it was asked."""

    def __init__(self):
        self.calls = []

    def loaded(self):
        return False

    def generate(self, **kw):
        self.calls.append(kw)
        ims = [Image.new("RGB", (32, 48), "white") for _ in range(kw["n"])]
        return ims, {"lora": kw["lora"], "params": {"seed": 30}, "files": ["outputs/x.png"]}


def test_api_serves_promoted_lora(root, tmp_path):
    from fastapi.testclient import TestClient

    from manga.api import create_app

    fake = FakeService()
    client = TestClient(create_app(service=fake, jobs=JobRunner(root), root=root, with_ui=False))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/series").json()[0]["name"] == "bungo"

    files = {"reference": open(_img(tmp_path / "r.png"), "rb"),
             "outline": open(_img(tmp_path / "o.png", (0, 0, 0)), "rb")}
    r = client.post("/generate", data={"series": "bungo", "description": "1boy"}, files=files)
    assert r.status_code == 409                                  # nothing promoted yet

    ck = registry.checkpoints("bungo", root)[0]
    registry.promote("bungo", ck, 0.7, root)
    for f in files.values():
        f.seek(0)
    r = client.post("/generate", data={"series": "bungo", "description": "1boy", "n": "2"},
                    files=files)
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert Image.open(__import__("io").BytesIO(r.content)).size == (64, 48)
    call = fake.calls[-1]
    assert call["lora"] == ck and call["lora_scale"] == 0.7 and call["trigger"] == "bsdstyle"
    assert len(library.items("refs", root)) == 1                 # inputs kept for reuse


def test_ui_builds(root):
    pytest.importorskip("gradio")
    from manga import ui
    from manga.service import ModelService

    jobs = JobRunner(root)
    demo = ui.build(ModelService(root=root, jobs=jobs), jobs, root)
    assert demo is not None
