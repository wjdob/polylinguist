from polylinguist.schemas import AddonSettings
from polylinguist.services.subtitles import parse_subtitle_text, prepare_translation_batch, render_dual_srt


def test_parse_and_render_dual_srt():
    raw = """1
00:00:01,000 --> 00:00:03,000
Hello there

2
00:00:04,000 --> 00:00:06,000
How are you?
"""
    cues = parse_subtitle_text(raw)
    assert len(cues) == 2
    rendered = render_dual_srt(
        cues,
        ["Hola", "Como estas?"],
        AddonSettings(format_mode="dual"),
    )
    assert "00:00:01,000 --> 00:00:03,000" in rendered
    assert "Hello there" in rendered
    assert "Hola" in rendered
    assert "<i>" not in rendered


def test_render_translated_only():
    raw = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello
"""
    cues = parse_subtitle_text(raw)
    rendered = render_dual_srt(
        cues,
        ["Bonjour"],
        AddonSettings(format_mode="translated_only"),
    )
    assert "Hello" not in rendered
    assert "Bonjour" in rendered


def test_prepare_translation_batch_sanitizes_junk_lines():
    prepared = prepare_translation_batch(
        [
            "<i>Hello</i>\n,,,,,,,,,,,,,,,,,,,,,,,,",
            "[wind whistling outside]",
            ",,,,,,,,,,,,,,,,,,,,,,,",
        ]
    )

    assert prepared.cues[0] == "Hello"
    assert prepared.active_cues == ["Hello", "[wind whistling outside]"]
    assert prepared.active_indices == [0, 1]
    assert prepared.sanitized_count == 2
    assert prepared.skipped_count == 1
