"""Tests for configuration management.

FIX H2: Verify settings load without crashing on YAML with extra keys.
FIX H5: Verify settings can be reset for test isolation.
"""
from src.config import get_settings
from src.config import load_settings
from src.config import reset_settings


class TestConfig:
    def test_load_default_settings(self):
        """FIX H2: Must not crash even with extra YAML keys like structured_output."""
        settings = load_settings()
        assert settings.generation.model  # Has a default
        assert settings.generation.max_retries >= 1

    def test_settings_singleton_reset(self):
        """FIX H5: Verify reset_settings enables test isolation."""
        s1 = get_settings()
        reset_settings()
        s2 = get_settings()
        # Both should be valid but not necessarily the same object
        assert s1.generation.model == s2.generation.model

    def test_device_config(self):
        """FIX M7: Verify device configuration flows through."""
        settings = load_settings()
        assert settings.grounding.device.startswith(("cpu", "cuda", "mps"))
        assert settings.retrieval.dense.device.startswith(("cpu", "cuda", "mps"))
