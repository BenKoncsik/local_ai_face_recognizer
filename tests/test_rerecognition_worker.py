"""Tests for ReRecognitionWorker."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.config import AppConfig
from app.jobs.cancellation import CancellationToken, OperationCancelled
from app.services.rerecognition_service import (
    KIND_AUTO,
    KIND_NONE,
    KIND_SUGGEST,
    AutoItem,
    FaceData,
    ReRecognitionResult,
    SuggestItem,
)
from app.workers.rerecognition_worker import ReRecognitionWorker


def _worker(image_ids=(1,), **kwargs) -> ReRecognitionWorker:
    return ReRecognitionWorker(list(image_ids), AppConfig(), **kwargs)


def _face(face_id: int = 1) -> FaceData:
    return FaceData(
        face_id=face_id,
        image_id=1,
        crop_path=None,
        embedding=np.zeros(128, dtype=np.float32),
        prev_person_id=2,
        prev_person_name="Unknown 1",
        prev_person_was_auto=True,
        prev_assignment_source="clustering",
        prev_assignment_confidence=None,
        prev_assigned_at=None,
    )


def _patch_service(monkeypatch, svc):
    @contextmanager
    def _scope():
        yield object()

    monkeypatch.setattr("app.workers.rerecognition_worker.session_scope", _scope)
    monkeypatch.setattr(
        "app.workers.rerecognition_worker.ReRecognitionService",
        lambda *a, **k: svc,
    )


class TestEmitProgress:
    def test_throttles_intermediate_updates(self, monkeypatch):
        w = _worker()
        w._last_emit = -1000.0  # allow the first progress tick through
        emitted = []
        w.progress.connect(lambda *a: emitted.append(a))
        times = iter([0.0, 0.0, 0.2])  # seconds → 0 ms, 0 ms, 200 ms
        monkeypatch.setattr(
            "app.workers.rerecognition_worker.time.monotonic",
            lambda: next(times),
        )
        w._emit_progress(1, 10, 0, 0)
        w._emit_progress(2, 10, 0, 0)  # throttled
        w._emit_progress(10, 10, 1, 0, force=True)
        assert len(emitted) == 2
        assert emitted[0] == (1, 10, 0, 0)
        assert emitted[-1] == (10, 10, 1, 0)

    def test_routes_through_task_context(self):
        w = _worker()
        ctx = MagicMock()
        ctx.token = MagicMock()
        w._ctx = ctx
        w._emit_progress(5, 10, 2, 1, force=True)
        ctx.report.assert_called_once_with(50, "5/10 (auto 2, ? 1)")
        ctx.token.wait_if_paused.assert_called_once()


class TestExecute:
    def test_nothing_to_do(self, monkeypatch):
        w = _worker()
        svc = MagicMock()
        svc.load_profiles.return_value = []
        svc.extract_candidates.return_value = []
        _patch_service(monkeypatch, svc)
        result = w._execute()
        assert result.n_examined == 0
        assert result.n_auto == 0

    def test_classifies_and_applies_auto(self, monkeypatch):
        w = _worker()
        face = _face()
        svc = MagicMock()
        svc.load_profiles.return_value = [object()]
        svc.load_rejected_targets.return_value = set()
        auto = AutoItem(face=face, target_person_id=9, target_name="Anna", score=0.95)
        svc.classify.side_effect = [
            (KIND_AUTO, auto),
            (KIND_SUGGEST, SuggestItem(face=face, candidates=[])),
            (KIND_NONE, None),
        ]
        svc.extract_candidates.return_value = [face, _face(2), _face(3)]
        _patch_service(monkeypatch, svc)
        result = w._execute()
        assert result.n_examined == 3
        assert result.n_auto == 1
        assert result.n_suggest == 1
        assert result.n_none == 1
        assert result.batch_id
        svc.apply_auto_merges.assert_called_once()

    def test_cancel_mid_loop(self, monkeypatch):
        w = _worker()
        w._token.cancel()
        svc = MagicMock()
        svc.load_profiles.return_value = [object()]
        svc.extract_candidates.return_value = [_face()]
        _patch_service(monkeypatch, svc)
        with pytest.raises(OperationCancelled):
            w._execute()


class TestRun:
    def test_run_emits_result(self, qtbot, monkeypatch):
        w = _worker()
        expected = ReRecognitionResult(n_examined=1, n_auto=1, batch_id="abc")
        w._execute = MagicMock(return_value=expected)
        w._run_ai_face_detection = MagicMock()
        with qtbot.waitSignal(w.finished_result, timeout=3000) as blocker:
            w.run()
        assert blocker.args[0] is expected

    def test_run_cancelled(self, qtbot, monkeypatch):
        w = _worker()
        w._execute = MagicMock(side_effect=OperationCancelled())
        with qtbot.waitSignal(w.finished_result, timeout=3000) as blocker:
            w.run()
        assert blocker.args[0].cancelled is True

    def test_run_failure_emits_failed(self, qtbot, monkeypatch):
        w = _worker()
        w._execute = MagicMock(side_effect=RuntimeError("boom"))
        with qtbot.waitSignal(w.failed, timeout=3000) as blocker:
            w.run()
        assert "boom" in blocker.args[0]

    def test_run_in_task_uses_ctx_token(self, monkeypatch):
        w = _worker()
        expected = ReRecognitionResult(n_examined=0)
        w._execute = MagicMock(return_value=expected)
        w._run_ai_face_detection = MagicMock()
        ctx = MagicMock()
        ctx.token = CancellationToken()
        result = w.run_in_task(ctx)
        assert result is expected
        assert w._token is ctx.token


class TestAiFaceDetection:
    def test_skipped_when_disabled(self):
        cfg = AppConfig()
        cfg.ai_face_detection.enabled = False
        cfg.ai_face_detection.run_on_rerecognition = False
        w = ReRecognitionWorker([1], cfg)
        result = ReRecognitionResult()
        w._run_ai_face_detection(result)
        assert result.ai_images_scanned is None

    def test_populates_stats_when_enabled(self, monkeypatch):
        cfg = AppConfig()
        cfg.ai_face_detection.enabled = True
        cfg.ai_face_detection.run_on_rerecognition = True
        w = ReRecognitionWorker([1], cfg)
        svc = MagicMock()
        stats = MagicMock(available=True, images_processed=2, faces_found=5, error=None)
        svc.detect_images.return_value = stats
        monkeypatch.setattr(
            "app.workers.rerecognition_worker.session_scope",
            lambda: MagicMock(__enter__=lambda s: svc, __exit__=MagicMock()),
        )
        with patch(
            "app.services.ai_face_detection_service.AiFaceDetectionService",
            return_value=svc,
        ):
            result = ReRecognitionResult()
            w._run_ai_face_detection(result)
        assert result.ai_images_scanned == 2
        assert result.ai_faces_found == 5
