from manga.config import load


def test_base_loads_and_has_frozen_parameters():
    cfg = load("base.yaml")
    assert cfg.generation.cn_scale == 0.50
    assert cfg.generation.cn_end == 0.70
    assert cfg.generation.ip_scale == 0.80
    assert cfg.model.clip_skip == 2


def test_artist_config_inherits_base():
    cfg = load("conan.yaml")
    assert cfg.model.checkpoint == load("base.yaml").model.checkpoint   # inherited
    assert cfg.artist.series == "detective_conan"                        # own
    assert len(cfg.poses) == 5
