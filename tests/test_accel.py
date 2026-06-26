"""Tests for app.accel — provider resolution and the status registry.

Pure logic, fully cross-platform: platform probes and the onnxruntime provider
list are monkeypatched so the same assertions hold on Windows, Linux and macOS.
"""

from __future__ import annotations

import pytest

from app import accel


@pytest.fixture(autouse=True)
def _clean_registry():
    accel.reset()
    yield
    accel.reset()


# ----------------------------------------------------------------------
# onnx_providers — platform-safe ordering
# ----------------------------------------------------------------------

class TestOnnxProviders:
    def test_cuda_keeps_priority_on_windows(self, monkeypatch):
        """A CUDA box (any OS) keeps CUDA first — unchanged behaviour."""
        monkeypatch.setattr(accel.platform, "system", lambda: "Windows")
        monkeypatch.setattr(
            accel, "_ort_available_providers",
            lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        assert accel.onnx_providers() == [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]

    def test_windows_cpu_only_unchanged(self, monkeypatch):
        """No CUDA, not macOS → plain CPU; CoreML is never added off macOS."""
        monkeypatch.setattr(accel.platform, "system", lambda: "Windows")
        monkeypatch.setattr(
            accel, "_ort_available_providers",
            lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        # Even if onnxruntime reports CoreML, it is ignored on Windows.
        assert accel.onnx_providers() == ["CPUExecutionProvider"]

    def test_macos_adds_coreml_before_cpu(self, monkeypatch):
        monkeypatch.setattr(accel.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            accel, "_ort_available_providers",
            lambda: ["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        assert accel.onnx_providers() == [
            "CoreMLExecutionProvider", "CPUExecutionProvider"
        ]

    def test_macos_cuda_still_wins(self, monkeypatch):
        """A (hypothetical) CUDA mac keeps CUDA ahead of CoreML."""
        monkeypatch.setattr(accel.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            accel, "_ort_available_providers",
            lambda: ["CUDAExecutionProvider", "CoreMLExecutionProvider",
                     "CPUExecutionProvider"],
        )
        assert accel.onnx_providers() == [
            "CUDAExecutionProvider", "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]

    def test_no_onnxruntime_falls_back_to_cpu(self, monkeypatch):
        monkeypatch.setattr(accel.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(accel, "_ort_available_providers", lambda: [])
        assert accel.onnx_providers() == ["CPUExecutionProvider"]


# ----------------------------------------------------------------------
# classify_providers
# ----------------------------------------------------------------------

class TestClassify:
    def test_cuda(self):
        assert accel.classify_providers(
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ) == accel.ACC_CUDA

    def test_coreml_partial_offload(self):
        # CPU first, CoreML still present → CoreML is doing work.
        assert accel.classify_providers(
            ["CPUExecutionProvider", "CoreMLExecutionProvider"]
        ) == accel.ACC_COREML

    def test_cpu_only(self):
        assert accel.classify_providers(["CPUExecutionProvider"]) == accel.ACC_CPU

    def test_empty(self):
        assert accel.classify_providers([]) == accel.ACC_CPU


# ----------------------------------------------------------------------
# Status registry
# ----------------------------------------------------------------------

class TestRegistry:
    def test_summary_empty(self):
        assert accel.summary() == "—"

    def test_summary_prefers_strongest(self):
        accel.report("embedding", accel.ACC_COREML, "ArcFace")
        accel.report("detection", accel.ACC_CPU, "SCRFD")
        # CoreML outranks CPU in the one-line summary.
        assert accel.summary() == accel.human(accel.ACC_COREML)

    def test_report_onnx_classifies(self):
        accel.report_onnx(
            "detection", ["CPUExecutionProvider", "CoreMLExecutionProvider"]
        )
        assert accel.active()["detection"]["acc"] == accel.ACC_COREML

    def test_detail_lines(self):
        accel.report("embedding", accel.ACC_COREML, "MobileFaceNet Core ML")
        lines = accel.detail_lines()
        assert any("embedding" in ln and "Neural Engine" in ln for ln in lines)
