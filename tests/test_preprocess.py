from manga.preprocess import Box


def test_box_overlap():
    a = Box(0, 0, 100, 100)
    assert a.overlap(Box(50, 50, 150, 150)) == 2500
    assert a.overlap(Box(200, 200, 300, 300)) == 0


def test_pad_is_clamped_to_image():
    b = Box(0, 0, 100, 100).pad(0.5, (120, 120))
    assert (b.xmin, b.ymin) == (0, 0)
    assert (b.xmax, b.ymax) == (120, 120)
