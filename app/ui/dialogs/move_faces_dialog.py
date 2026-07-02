"""Dialog for choosing the destination of a batch face move.

Thin wrapper around the shared
:class:`app.ui.dialogs.person_picker_dialog.PersonPickerDialog`: pick an
existing person (searchable list with face-match percentages) or type a name to
create a brand-new person for the selected faces.  The source person is
excluded from the list so faces cannot be "moved" onto themselves.

Match scores can be supplied pre-computed (*match_scores*) or — preferred on
large databases — computed in the background by passing *score_face_ids*: the
dialog opens instantly with a "computing similarity…" hint and re-ranks itself
when scoring finishes.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from PySide6.QtWidgets import QWidget

from app.db.models import Person
from app.ui.dialogs.person_picker_dialog import PersonPickerDialog
from app.ui.i18n import t


class MoveFacesDialog(PersonPickerDialog):
    """Pick an existing person or name a new one as the move destination."""

    def __init__(
        self,
        face_count: int,
        persons: List[Person],
        exclude_person_id: Optional[int] = None,
        match_scores: Optional[Dict[int, float]] = None,
        score_face_ids: Optional[Sequence[int]] = None,
        recognition_cfg=None,
        config=None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(
            persons,
            title=t("move_faces_title"),
            header=t("move_faces_header", n=face_count),
            exclude_person_id=exclude_person_id,
            match_scores=match_scores,
            score_face_ids=score_face_ids,
            recognition_cfg=recognition_cfg,
            config=config,
            allow_create_new=True,
            ok_label=t("move_faces_confirm"),
            parent=parent,
        )
