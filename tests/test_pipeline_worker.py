"""Tests for PipelineWorker and PipelineResult."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.config import AppConfig
from app.jobs.cancellation import CancellationToken, OperationCancelled
from app.services.clustering_service import ClusteringStats
from app.services.intra_image_consistency_service import IntraImageConsistencyStats
from app.services.intra_image_duplicate_service import IntraImageDuplicateStats
from app.services.recognition_service import RecognitionStats
from app.workers.pipeline_worker import PipelineResult, PipelineWorker


def _worker(**kwargs) -> PipelineWorker:
    return PipelineWorker(root_folder="/photos", config=AppConfig(), **kwargs)


class TestPipelineResult:
    def test_defaults(self):
        r = PipelineResult(success=True, summary="ok")
        assert r.n_suggestions == 0
        assert r.n_auto_assignments == 0

    def test_fields(self):
        r = PipelineResult(False, "failed", n_suggestions=3, n_auto_assignments=1)
        assert r.success is False
        assert r.summary == "failed"
        assert r.n_suggestions == 3
        assert r.n_auto_assignments == 1


class TestPipelineWorkerHelpers:
    def test_drive_mode_requires_client(self):
        w = _worker()
        assert w._drive_mode is False
        w = _worker(drive_client=object())
        assert w._drive_mode is True

    def test_abort_sets_flag(self):
        w = _worker()
        w.abort()
        with pytest.raises(OperationCancelled):
            w._checkpoint()

    def test_emit_progress_legacy_signal(self, qtbot):
        w = _worker()
        with qtbot.waitSignal(w.progress, timeout=1000) as blocker:
            w._emit_progress(5, 10, "Scanning", "img.jpg")
        assert list(blocker.args) == [5, 10, "Scanning", "img.jpg"]

    def test_emit_progress_task_context(self):
        w = _worker()
        ctx = MagicMock()
        ctx.report = MagicMock()
        ctx.checkpoint = MagicMock()
        w._ctx = ctx
        w._emit_progress(25, 100, "Embedding", "face #1")
        ctx.report.assert_called_once_with(25, "Embedding: face #1")
        ctx.checkpoint.assert_called_once()

    def test_emit_log_emits_signal(self, qtbot):
        w = _worker()
        with qtbot.waitSignal(w.log_message, timeout=1000) as blocker:
            w._emit_log("hello")
        assert list(blocker.args) == ["hello"]

    def test_checkpoint_task_context(self):
        w = _worker()
        ctx = MagicMock()
        w._ctx = ctx
        w._checkpoint()
        ctx.checkpoint.assert_called_once()


class TestRunPipeline:
    def _patch_stages(self, worker: PipelineWorker):
        worker._run_scan = MagicMock(return_value=[1, 2])
        worker._get_pending_detection_ids = MagicMock(return_value=[10])
        worker._run_detection = MagicMock(return_value=4)
        worker._run_embedding = MagicMock(return_value=3)
        worker._run_intra_image_dedup = MagicMock(
            return_value=IntraImageDuplicateStats(faces_removed=1)
        )
        worker._run_recognition = MagicMock(
            return_value=RecognitionStats(
                n_assigned=2, n_candidates=5, n_profiles=3,
                n_below_threshold=1, n_margin_rejected=0,
            )
        )
        worker._run_ignored_filter = MagicMock(
            return_value=MagicMock(n_suppressed=0)
        )
        worker._run_clustering = MagicMock(
            return_value=ClusteringStats(
                n_new_persons=1, n_assigned_to_new=2,
                n_assigned_to_existing=0, n_singletons=1,
            )
        )
        worker._run_intra_image_consistency = MagicMock(
            return_value=IntraImageConsistencyStats(
                n_faces_reassigned=1, n_persons_removed=0,
            )
        )
        worker._run_suggestions = MagicMock(return_value=2)

    def test_run_pipeline_success(self, tmp_path, monkeypatch):
        w = _worker(db_path_override=str(tmp_path / "pipe.db"))
        self._patch_stages(w)
        fake_qs = MagicMock()
        fake_qs.value.return_value = True
        monkeypatch.setattr(
            "app.app_settings.app_qsettings", lambda: fake_qs
        )
        monkeypatch.setattr("app.workers.pipeline_worker.init_db", MagicMock())

        result = w._run_pipeline()

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.n_suggestions == 2
        assert "2 new image(s)" in result.summary
        assert "4 detected" in result.summary

    def test_run_legacy_emits_finished(self, qtbot, tmp_path, monkeypatch):
        w = _worker(db_path_override=str(tmp_path / "pipe.db"))
        self._patch_stages(w)
        fake_qs = MagicMock()
        fake_qs.value.return_value = False
        monkeypatch.setattr("app.app_settings.app_qsettings", lambda: fake_qs)
        monkeypatch.setattr("app.workers.pipeline_worker.init_db", MagicMock())

        with qtbot.waitSignal(w.suggestions_ready, timeout=3000) as sug_blocker:
            with qtbot.waitSignal(w.finished, timeout=3000) as fin_blocker:
                w.run()
        assert list(sug_blocker.args) == [2]
        assert list(fin_blocker.args)[0] is True

    def test_run_legacy_cancelled(self, qtbot):
        w = _worker()
        w._run_pipeline = MagicMock(side_effect=OperationCancelled())
        with qtbot.waitSignal(w.finished, timeout=3000) as blocker:
            w.run()
        assert list(blocker.args) == [False, "Aborted"]

    def test_run_in_task_returns_result(self, tmp_path, monkeypatch):
        w = _worker(db_path_override=str(tmp_path / "pipe.db"))
        self._patch_stages(w)
        fake_qs = MagicMock()
        fake_qs.value.return_value = True
        monkeypatch.setattr("app.app_settings.app_qsettings", lambda: fake_qs)
        monkeypatch.setattr("app.workers.pipeline_worker.init_db", MagicMock())

        ctx = MagicMock()
        ctx.token = CancellationToken()
        result = w.run_in_task(ctx)
        assert result.success is True
        assert w._ctx is ctx
