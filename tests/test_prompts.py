import re

from manga import prompts
from manga.config import load

COLOR_WORDS = r"\b(red|blue|green|yellow|purple|orange|pink|brown|blonde|crimson)\b"


def test_prompt_contains_style_character_and_pose():
    cfg = load("conan.yaml")
    p = prompts.character_prompt(cfg, "full body, walking")
    assert "monochrome" in p
    assert "bow tie" in p
    assert "full body, walking" in p


def test_no_color_words_anywhere():
    """Color words drag a monochrome prompt back toward color. POC finding."""
    cfg = load("conan.yaml")
    p = prompts.character_prompt(cfg, "full body, standing")
    assert not re.search(COLOR_WORDS, p, re.I), f"color word in prompt: {p}"


def test_reference_props_are_negated():
    cfg = load("conan.yaml")
    assert "bag" in prompts.character_negative(cfg)
