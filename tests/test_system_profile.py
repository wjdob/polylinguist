from polylinguist.services.system_profile import (
    _VideoAdapter,
    _detect_accelerators,
    _detect_free_disk_gb,
    _python_supports_openvino_gpu,
)


def test_detect_accelerators_exposes_intel_openvino_without_runtime(monkeypatch):
    monkeypatch.setattr(
        "polylinguist.services.system_profile._detect_windows_video_adapters",
        lambda: [_VideoAdapter(vendor="intel", name="Intel(R) Arc(TM) A770 Graphics")],
    )
    monkeypatch.setattr("polylinguist.services.system_profile._detect_nvidia_gpu_names", lambda: [])
    monkeypatch.setattr("polylinguist.services.system_profile._python_supports_openvino_gpu", lambda: True)

    accelerators = list(_detect_accelerators("windows"))

    assert len(accelerators) == 1
    assert accelerators[0].vendor == "intel"
    assert accelerators[0].supported_targets == ("openvino_gpu",)


def test_detect_accelerators_skips_non_arc_intel_adapters(monkeypatch):
    monkeypatch.setattr(
        "polylinguist.services.system_profile._detect_windows_video_adapters",
        lambda: [_VideoAdapter(vendor="intel", name="Intel(R) UHD Graphics 770")],
    )
    monkeypatch.setattr("polylinguist.services.system_profile._detect_nvidia_gpu_names", lambda: [])
    monkeypatch.setattr("polylinguist.services.system_profile._python_supports_openvino_gpu", lambda: True)

    accelerators = list(_detect_accelerators("windows"))

    assert accelerators == []


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
    monkeypatch.setattr("polylinguist.services.system_profile._python_supports_openvino_gpu", lambda: True)

    accelerators = list(_detect_accelerators("windows"))

    assert len(accelerators) == 1
    assert accelerators[0].name == "Intel(R) Arc(TM) A370M Graphics"


def test_detect_free_disk_gb_falls_back_when_home_is_blocked(monkeypatch):
    calls = []

    def fake_disk_usage(path):
        calls.append(path)
        if len(calls) == 1:
            raise PermissionError("blocked")
        return type("Usage", (), {"free": 30 * 1024 ** 3})()

    monkeypatch.setattr("polylinguist.services.system_profile.shutil.disk_usage", fake_disk_usage)
    monkeypatch.setattr("polylinguist.services.system_profile.os.getcwd", lambda: r"C:\workspace")
    monkeypatch.setattr("polylinguist.services.system_profile.os.path.expanduser", lambda value: r"C:\Users\blocked")
    monkeypatch.delenv("POLYLINGUIST_HOME", raising=False)

    free_disk_gb = _detect_free_disk_gb()

    assert free_disk_gb == 30.0
    assert calls[0] == r"C:\Users\blocked"


def test_python_314_disables_openvino_gpu_support(monkeypatch):
    monkeypatch.setattr("polylinguist.services.system_profile.platform.system", lambda: "Windows")
    monkeypatch.setattr("polylinguist.services.system_profile.sys.version_info", (3, 14, 1))

    assert _python_supports_openvino_gpu() is False
