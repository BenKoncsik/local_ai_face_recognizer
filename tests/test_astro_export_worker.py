"""Tests for AstroExportWorker."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.workers.astro_export_worker import AstroExportWorker


@contextmanager
def _session_scope():
    yield object()


class TestAstroExportWorker:
    def test_emits_finished_ok(self, qtbot, tmp_path):
        out = tmp_path / "site"
        with patch(
            "app.workers.astro_export_worker.ExportService"
        ) as SvcCls, patch(
            "app.workers.astro_export_worker.session_scope", _session_scope
        ):
            SvcCls.return_value.export_astro.return_value = out
            worker = AstroExportWorker(str(tmp_path / "target"), person_id=5)
            with qtbot.waitSignal(worker.finished_ok, timeout=3000) as blocker:
                worker.run()
        assert blocker.args[0] == str(out)
        SvcCls.return_value.export_astro.assert_called_once()
        kwargs = SvcCls.return_value.export_astro.call_args.kwargs
        assert kwargs["person_id"] == 5

    def test_reports_progress(self, qtbot, tmp_path):
        seen = []

        def _export(**kwargs):
            cb = kwargs["progress_callback"]
            cb(42, "resizing")
            return tmp_path / "out"

        with patch(
            "app.workers.astro_export_worker.ExportService"
        ) as SvcCls, patch(
            "app.workers.astro_export_worker.session_scope", _session_scope
        ):
            SvcCls.return_value.export_astro.side_effect = _export
            worker = AstroExportWorker(str(tmp_path / "target"))
            worker.progress.connect(lambda *a: seen.append(a))
            worker.run()

        assert seen == [(42, "resizing")]

    def test_node_error_kind(self, qtbot, tmp_path):
        with patch(
            "app.workers.astro_export_worker.ExportService"
        ) as SvcCls, patch(
            "app.workers.astro_export_worker.session_scope", _session_scope
        ):
            SvcCls.return_value.export_astro.side_effect = RuntimeError(
                "npm not found"
            )
            worker = AstroExportWorker(str(tmp_path / "target"))
            with qtbot.waitSignal(worker.failed, timeout=3000) as blocker:
                worker.run()
        assert list(blocker.args) == ["node", "npm not found"]

    def test_generic_error_kind(self, qtbot, tmp_path):
        with patch(
            "app.workers.astro_export_worker.ExportService"
        ) as SvcCls, patch(
            "app.workers.astro_export_worker.session_scope", _session_scope
        ):
            SvcCls.return_value.export_astro.side_effect = ValueError("bad data")
            worker = AstroExportWorker(str(tmp_path / "target"))
            with qtbot.waitSignal(worker.failed, timeout=3000) as blocker:
                worker.run()
        assert blocker.args[0] == "error"
