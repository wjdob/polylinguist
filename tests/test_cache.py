from polylinguist.services.cache import SubtitleCache


def test_subtitle_cache_preserves_line_endings(tmp_path):
    cache = SubtitleCache(tmp_path)
    content = "1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\nWorld\r\n"

    cache.put("example-key", content)
    restored = cache.get("example-key")

    assert restored == content
    assert "\r\r\n" not in restored
