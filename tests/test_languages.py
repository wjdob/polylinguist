from polylinguist.services.languages import get_language, normalize_language


def test_normalize_language_aliases():
    assert normalize_language("fra") == "fre"
    assert normalize_language("deu") == "ger"
    assert normalize_language("eng") == "eng"


def test_get_language_has_backend_codes():
    spec = get_language("zht")
    assert spec is not None
    assert spec.iso639_1 == "zh"
    assert spec.nllb_code == "zho_Hant"
