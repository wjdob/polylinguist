from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageSpec:
    canonical: str
    label: str
    iso639_1: str | None
    marian_code: str | None
    nllb_code: str | None
    aliases: tuple[str, ...] = ()


LANGUAGE_SPECS: tuple[LanguageSpec, ...] = (
    LanguageSpec("eng", "English", "en", "en", "eng_Latn"),
    LanguageSpec("spa", "Spanish", "es", "es", "spa_Latn"),
    LanguageSpec("fre", "French", "fr", "fr", "fra_Latn", ("fra",)),
    LanguageSpec("ger", "German", "de", "de", "deu_Latn", ("deu",)),
    LanguageSpec("ita", "Italian", "it", "it", "ita_Latn"),
    LanguageSpec("por", "Portuguese", "pt", "pt", "por_Latn"),
    LanguageSpec("pob", "Portuguese (Brazil)", "pt", "pt", "por_Latn"),
    LanguageSpec("tur", "Turkish", "tr", "tr", "tur_Latn"),
    LanguageSpec("rus", "Russian", "ru", "ru", "rus_Cyrl"),
    LanguageSpec("ukr", "Ukrainian", "uk", "uk", "ukr_Cyrl"),
    LanguageSpec("pol", "Polish", "pl", "pl", "pol_Latn"),
    LanguageSpec("dut", "Dutch", "nl", "nl", "nld_Latn", ("nld",)),
    LanguageSpec("swe", "Swedish", "sv", "sv", "swe_Latn"),
    LanguageSpec("cze", "Czech", "cs", "cs", "ces_Latn", ("ces",)),
    LanguageSpec("jpn", "Japanese", "ja", "ja", "jpn_Jpan"),
    LanguageSpec("kor", "Korean", "ko", "ko", "kor_Hang"),
    LanguageSpec("chi", "Chinese (Simplified)", "zh", "zh", "zho_Hans", ("zho",)),
    LanguageSpec("zht", "Chinese (Traditional)", "zh", "zh", "zho_Hant"),
    LanguageSpec("ara", "Arabic", "ar", "ar", "arb_Arab"),
    LanguageSpec("hin", "Hindi", "hi", "hi", "hin_Deva"),
)


LANGUAGES_BY_CODE = {
    alias: spec
    for spec in LANGUAGE_SPECS
    for alias in (spec.canonical, *spec.aliases)
}


def normalize_language(code: str | None) -> str | None:
    if not code:
        return None
    lowered = code.strip().lower()
    spec = LANGUAGES_BY_CODE.get(lowered)
    if spec:
        return spec.canonical
    return lowered


def get_language(code: str | None) -> LanguageSpec | None:
    normalized = normalize_language(code)
    if normalized is None:
        return None
    return LANGUAGES_BY_CODE.get(normalized)


def list_languages() -> list[LanguageSpec]:
    return list(LANGUAGE_SPECS)


def language_label(code: str | None) -> str:
    spec = get_language(code)
    if spec:
        return spec.label
    return (code or "Unknown").upper()
