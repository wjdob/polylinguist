from polylinguist.services.system_profile import _VideoAdapter, _detect_accelerators


def test_detect_accelerators_exposes_intel_openvino_without_runtime(monkeypatch):
    monkeypatch.setattr(
        "polylinguist.services.system_profile._detect_windows_video_adapters",
        lambda: [_VideoAdapter(vendor="intel", name="Intel(R) Arc(TM) Graphics")],
    )
    monkeypatch.setattr("polylinguist.services.system_profile._detect_nvidia_gpu_names", lambda: [])

    accelerators = list(_detect_accelerators("windows"))

    assert len(accelerators) == 1
    assert accelerators[0].vendor == "intel"
    assert accelerators[0].supported_targets == ("openvino_gpu",)


def test_detect_accelerators_exposes_amd_directml_without_runtime(monkeypatch):
    monkeypatch.setattr(
        "polylinguist.services.system_profile._detect_windows_video_adapters",
        lambda: [_VideoAdapter(vendor="amd", name="AMD Radeon Graphics")],
    )
    monkeypatch.setattr("polylinguist.services.system_profile._detect_nvidia_gpu_names", lambda: [])

    accelerators = list(_detect_accelerators("windows"))

    assert len(accelerators) == 1
    assert accelerators[0].vendor == "amd"
    assert accelerators[0].supported_targets == ("directml",)


def test_detect_accelerators_filters_windows_virtual_adapters(monkeypatch):
    monkeypatch.setattr(
        "polylinguist.services.system_profile._detect_windows_video_adapters",
        lambda: [
            _VideoAdapter(vendor="intel", name="Microsoft Basic Display Adapter"),
            _VideoAdapter(vendor="intel", name="Intel(R) Arc(TM) A370M Graphics"),
        ],
    )
    monkeypatch.setattr("polylinguist.services.system_profile._detect_nvidia_gpu_names", lambda: [])

    accelerators = list(_detect_accelerators("windows"))

    assert len(accelerators) == 1
    assert accelerators[0].name == "Intel(R) Arc(TM) A370M Graphics"
