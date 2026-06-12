import sys
from threading import Lock
from types import ModuleType, SimpleNamespace

from polylinguist.model_worker import translate_nllb
from polylinguist.services.translation import NLLB_MAX_NEW_TOKENS, NllbTranslator, TranslationRequest


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorchModule:
    @staticmethod
    def no_grad():
        return _NoGrad()


class _FakeTokenizer:
    def __init__(self) -> None:
        self.decode_count = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "zho_Hans"
        return 7

    def __call__(self, batch, return_tensors: str, padding: bool, truncation: bool):
        return {"input_ids": list(batch), "attention_mask": [1] * len(batch)}

    def batch_decode(self, generated, skip_special_tokens: bool = True):
        self.decode_count += 1
        return ["translated"] * len(generated)


class _FakeModel:
    def __init__(self) -> None:
        self.calls = []
        self.generation_config = SimpleNamespace(max_length=None)

    def eval(self) -> None:
        return None

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = len(kwargs["input_ids"])
        return [[101, 102]] * batch_size


class _FakeRegistry:
    def metadata_for(self, provider: str, model_id: str):
        return {}


def test_active_nllb_translation_sets_max_new_tokens(monkeypatch):
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    translator = NllbTranslator(_FakeRegistry())

    monkeypatch.setattr("polylinguist.services.translation._ensure_python_packages", lambda *args, **kwargs: None)
    monkeypatch.setattr("polylinguist.services.translation._current_runtime_has_cuda", lambda: False)
    monkeypatch.setattr(
        translator,
        "_get_bundle",
        lambda model_id, device, progress=None: {"tokenizer": tokenizer, "model": model, "lock": Lock()},
    )
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule())

    request = TranslationRequest(
        provider="nllb",
        model_id="facebook/nllb-200-distilled-600M",
        source_lang="eng",
        target_lang="chi",
        device_preference="cpu",
    )

    translated = translator.translate_batch(request, ["Keep your eyes on the queen, right?"])

    assert translated == ["translated"]
    assert len(model.calls) == 1
    assert model.calls[0]["forced_bos_token_id"] == 7
    assert model.calls[0]["max_new_tokens"] == NLLB_MAX_NEW_TOKENS


def test_worker_nllb_translation_sets_max_new_tokens(monkeypatch):
    tokenizer = _FakeTokenizer()
    model = _FakeModel()
    fake_transformers = ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str):
            return tokenizer

    class _FakeAutoModelForSeq2SeqLM:
        @staticmethod
        def from_pretrained(model_id: str):
            return model

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    fake_transformers.AutoModelForSeq2SeqLM = _FakeAutoModelForSeq2SeqLM

    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("polylinguist.model_worker._resolve_device_preference", lambda preference: "cpu")
    monkeypatch.setattr("polylinguist.model_worker.emit", lambda event, message: None)

    translated = translate_nllb(
        "facebook/nllb-200-distilled-600M",
        "eng",
        "chi",
        ["Keep your eyes on the queen, right?"],
        "cpu",
    )

    assert translated == ["translated"]
    assert len(model.calls) == 1
    assert model.calls[0]["forced_bos_token_id"] == 7
    assert model.calls[0]["max_new_tokens"] == NLLB_MAX_NEW_TOKENS
