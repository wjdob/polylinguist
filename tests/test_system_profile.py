from polylinguist.services.system_profile import SystemProfile


def test_profile_tiers():
    assert SystemProfile("windows", "amd64", 2, 4.0, 30.0, False, False).tier == "low"
    assert SystemProfile("windows", "amd64", 6, 12.0, 30.0, False, False).tier == "standard"
    assert SystemProfile("windows", "amd64", 8, 24.0, 30.0, True, False).tier == "strong"
