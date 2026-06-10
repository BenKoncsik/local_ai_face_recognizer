"""Tests for deoldified image pairing — filename parsing and DB lookup."""

from __future__ import annotations

import pytest

from app.services.deoldified_pairing_service import (
    ComparisonMember,
    DeoldifiedPairingService,
    extract_original_filename,
    extract_original_stem,
    extract_variant_label,
    is_deoldified_path,
)

# ──────────────────────────────────────────────────────────────────────────────
# Pure filename-parsing tests (no DB required)
# ──────────────────────────────────────────────────────────────────────────────


class TestExtractOriginalStem:
    def test_artistic_suffix(self) -> None:
        assert extract_original_stem("kép-deoldified (artistic)") == "kép"

    def test_stable_suffix(self) -> None:
        assert extract_original_stem("kép-deoldified (stable)") == "kép"

    def test_no_extra_suffix(self) -> None:
        assert extract_original_stem("kép-deoldified") == "kép"

    def test_not_deoldified_returns_none(self) -> None:
        assert extract_original_stem("normal_photo") is None

    def test_case_insensitive_upper(self) -> None:
        assert extract_original_stem("kép-DEOLDIFIED (artistic)") == "kép"

    def test_case_insensitive_mixed(self) -> None:
        assert extract_original_stem("kép-Deoldified") == "kép"

    def test_complex_name_from_requirements(self) -> None:
        stem = (
            "1984 [1984 Szemes (Ff neg Sf1_14)] PICT0346 (Kórus)-deoldified (artistic)"
        )
        expected = "1984 [1984 Szemes (Ff neg Sf1_14)] PICT0346 (Kórus)"
        assert extract_original_stem(stem) == expected

    def test_empty_stem_returns_none(self) -> None:
        assert extract_original_stem("-deoldified (artistic)") is None

    def test_stem_with_spaces(self) -> None:
        assert extract_original_stem("my photo-deoldified") == "my photo"


class TestExtractOriginalFilename:
    def test_jpg_uppercase_artistic(self) -> None:
        name = "kép-deoldified (artistic).JPG"
        assert extract_original_filename(name) == "kép.JPG"

    def test_jpg_lowercase_stable(self) -> None:
        name = "kép-deoldified (stable).jpg"
        assert extract_original_filename(name) == "kép.jpg"

    def test_no_extra_suffix(self) -> None:
        assert extract_original_filename("kép-deoldified.jpg") == "kép.jpg"

    def test_not_deoldified_returns_none(self) -> None:
        assert extract_original_filename("normal.jpg") is None

    def test_complex_name_from_requirements(self) -> None:
        name = (
            "1984 [1984 Szemes (Ff neg Sf1_14)] PICT0346 (Kórus)"
            "-deoldified (artistic).JPG"
        )
        expected = (
            "1984 [1984 Szemes (Ff neg Sf1_14)] PICT0346 (Kórus).JPG"
        )
        assert extract_original_filename(name) == expected

    def test_valami_stable(self) -> None:
        assert extract_original_filename("valami-deoldified (stable).jpg") == "valami.jpg"

    def test_jpeg_extension(self) -> None:
        assert extract_original_filename("foto-deoldified.jpeg") == "foto.jpeg"

    def test_png_extension(self) -> None:
        assert extract_original_filename("foto-deoldified.png") == "foto.png"

    def test_preserves_original_extension_case(self) -> None:
        # Extension case must be preserved, not changed
        assert extract_original_filename("foto-deoldified.JPEG") == "foto.JPEG"


class TestIsDeoldifiedPath:
    def test_simple_deoldified_path(self) -> None:
        assert is_deoldified_path("/some/folder/kép-deoldified.jpg") is True

    def test_deoldified_path_with_artistic(self) -> None:
        assert is_deoldified_path("/folder/kép-deoldified (artistic).JPG") is True

    def test_normal_path_returns_false(self) -> None:
        assert is_deoldified_path("/folder/normal.jpg") is False

    def test_filename_only_deoldified(self) -> None:
        assert is_deoldified_path("photo-deoldified.jpg") is True

    def test_filename_only_normal(self) -> None:
        assert is_deoldified_path("photo.jpg") is False

    def test_case_insensitive(self) -> None:
        assert is_deoldified_path("photo-DEOLDIFIED.jpg") is True


# ──────────────────────────────────────────────────────────────────────────────
# DB-backed tests for DeoldifiedPairingService
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path):
    from app.db.database import init_db
    db_file = tmp_path / "test.db"
    init_db(db_file)
    return db_file


class TestFindOriginalForDeoldified:
    def test_finds_original_exact_extension(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/folder/photo.jpg",
                file_hash="orig",
                file_mtime=0.0,
            )
            color = Image(
                file_path="/folder/photo-deoldified (artistic).jpg",
                file_hash="color",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            color_img = s.query(Image).filter(
                Image.file_hash == "color"
            ).first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_original_for_deoldified(color_img)
            assert result is not None
            assert result.file_hash == "orig"

    def test_finds_original_uppercase_extension_variant(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/folder/photo.JPG",
                file_hash="orig_up",
                file_mtime=0.0,
            )
            color = Image(
                file_path="/folder/photo-deoldified (artistic).JPG",
                file_hash="color_up",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            color_img = s.query(Image).filter(
                Image.file_hash == "color_up"
            ).first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_original_for_deoldified(color_img)
            assert result is not None
            assert result.file_hash == "orig_up"

    def test_returns_none_when_original_not_in_db(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            color = Image(
                file_path="/folder/photo-deoldified.jpg",
                file_hash="color_only",
                file_mtime=0.0,
            )
            s.add(color)

        with session_scope() as s:
            color_img = s.query(Image).filter(
                Image.file_hash == "color_only"
            ).first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_original_for_deoldified(color_img)
            assert result is None

    def test_finds_original_in_different_folder(self, tmp_db) -> None:
        """Original in a different folder must match by filename only."""
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/bw_folder/photo.JPG",
                file_hash="orig_xfolder",
                file_mtime=0.0,
            )
            color = Image(
                file_path="/color_folder/photo-deoldified (artistic).JPG",
                file_hash="color_xfolder",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            color_img = s.query(Image).filter(
                Image.file_hash == "color_xfolder"
            ).first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_original_for_deoldified(color_img)
            assert result is not None
            assert result.file_hash == "orig_xfolder"

    def test_returns_none_for_non_deoldified_image(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            img = Image(
                file_path="/folder/photo.jpg",
                file_hash="plain",
                file_mtime=0.0,
            )
            s.add(img)

        with session_scope() as s:
            plain = s.query(Image).filter(Image.file_hash == "plain").first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_original_for_deoldified(plain)
            assert result is None


class TestFindDeoldifiedForOriginal:
    def test_finds_colorized_variant(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/folder/photo.jpg",
                file_hash="orig2",
                file_mtime=0.0,
            )
            color = Image(
                file_path="/folder/photo-deoldified (stable).jpg",
                file_hash="color2",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            orig_img = s.query(Image).filter(Image.file_hash == "orig2").first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_deoldified_for_original(orig_img)
            assert result is not None
            assert result.file_hash == "color2"

    def test_returns_none_when_no_colorized_in_db(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/folder/photo.jpg",
                file_hash="orig_alone",
                file_mtime=0.0,
            )
            s.add(orig)

        with session_scope() as s:
            orig_img = s.query(Image).filter(
                Image.file_hash == "orig_alone"
            ).first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_deoldified_for_original(orig_img)
            assert result is None

    def test_different_folder_is_matched(self, tmp_db) -> None:
        """Colorized image in a different folder must match by filename."""
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/folderA/photo.jpg",
                file_hash="orig_a",
                file_mtime=0.0,
            )
            color = Image(
                file_path="/folderB/photo-deoldified.jpg",
                file_hash="color_b",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            orig_img = s.query(Image).filter(Image.file_hash == "orig_a").first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_deoldified_for_original(orig_img)
            assert result is not None
            assert result.file_hash == "color_b"

    def test_different_stem_not_matched(self, tmp_db) -> None:
        """A deoldified image with a different stem must not match."""
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/folderA/photo.jpg",
                file_hash="orig_diff",
                file_mtime=0.0,
            )
            color = Image(
                file_path="/folderB/otherphoto-deoldified.jpg",
                file_hash="color_diff",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            orig_img = s.query(Image).filter(
                Image.file_hash == "orig_diff"
            ).first()
            svc = DeoldifiedPairingService(s)
            result = svc.find_deoldified_for_original(orig_img)
            assert result is None


class TestPairingWithFaceData:
    """Verify that face data from the original is accessible via the pairing service."""

    def test_original_has_faces_colorized_has_none(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Face, Image, Person

        with session_scope() as s:
            orig = Image(
                file_path="/folder/portrait.jpg",
                file_hash="o_face",
                file_mtime=0.0,
                detection_done=True,
            )
            color = Image(
                file_path="/folder/portrait-deoldified.jpg",
                file_hash="c_face",
                file_mtime=0.0,
                detection_done=False,
            )
            s.add_all([orig, color])
            s.flush()

            person = Person(name="Test Person", is_auto_named=False)
            s.add(person)
            s.flush()

            face = Face(
                image_id=orig.id,
                person_id=person.id,
                bbox_x=10,
                bbox_y=20,
                bbox_w=80,
                bbox_h=80,
                confidence=0.9,
                detector_backend="cpu",
            )
            s.add(face)

        with session_scope() as s:
            color_img = s.query(Image).filter(Image.file_hash == "c_face").first()
            svc = DeoldifiedPairingService(s)
            original = svc.find_original_for_deoldified(color_img)
            assert original is not None
            assert len(original.faces) == 1
            assert original.faces[0].person.name == "Test Person"
            # colorized image itself has no faces
            assert len(color_img.faces) == 0


def _make_face(image_id: int, person_id: int):
    from app.db.models import Face

    return Face(
        image_id=image_id,
        person_id=person_id,
        bbox_x=10,
        bbox_y=20,
        bbox_w=80,
        bbox_h=80,
        confidence=0.9,
        detector_backend="cpu",
        assignment_source="manual",
    )


class TestSyncPairData:
    """One-directional copy of annotations between a deoldified pair."""

    def test_copies_faces_and_metadata_into_empty_side(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image, Person

        with session_scope() as s:
            orig = Image(
                file_path="/bw/photo.jpg",
                file_hash="src_full",
                file_mtime=0.0,
                detection_done=True,
                embedding_done=True,
                photo_date="1984",
                note="Kórus",
            )
            color = Image(
                file_path="/color/photo-deoldified (artistic).jpg",
                file_hash="dst_empty",
                file_mtime=0.0,
            )
            s.add_all([orig, color])
            s.flush()
            person = Person(name="Anna", is_auto_named=False)
            s.add(person)
            s.flush()
            s.add(_make_face(orig.id, person.id))

        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "src_full").first()
            color = s.query(Image).filter(Image.file_hash == "dst_empty").first()
            svc = DeoldifiedPairingService(s)
            result = svc.sync_pair_data(color, orig)  # order must not matter

            assert result is not None
            assert result["source_id"] == orig.id
            assert result["target_id"] == color.id
            assert result["faces_copied"] == 1
            assert "photo_date" in result["metadata_fields"]
            assert "note" in result["metadata_fields"]

        with session_scope() as s:
            color = s.query(Image).filter(Image.file_hash == "dst_empty").first()
            assert len(color.faces) == 1
            assert color.faces[0].person.name == "Anna"
            assert color.faces[0].assignment_source == "manual"
            assert color.photo_date == "1984"
            assert color.note == "Kórus"
            assert color.detection_done is True
            assert color.embedding_done is True
            # source is unchanged
            orig = s.query(Image).filter(Image.file_hash == "src_full").first()
            assert len(orig.faces) == 1

    def test_skips_when_both_sides_have_data(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image, Person

        with session_scope() as s:
            orig = Image(
                file_path="/bw/p.jpg", file_hash="b_full", file_mtime=0.0
            )
            color = Image(
                file_path="/color/p-deoldified.jpg",
                file_hash="c_full",
                file_mtime=0.0,
            )
            s.add_all([orig, color])
            s.flush()
            p = Person(name="X", is_auto_named=False)
            s.add(p)
            s.flush()
            s.add(_make_face(orig.id, p.id))
            s.add(_make_face(color.id, p.id))

        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "b_full").first()
            color = s.query(Image).filter(Image.file_hash == "c_full").first()
            svc = DeoldifiedPairingService(s)
            assert svc.sync_pair_data(orig, color) is None

    def test_skips_when_both_sides_empty(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/bw/e.jpg", file_hash="b_empty", file_mtime=0.0
            )
            color = Image(
                file_path="/color/e-deoldified.jpg",
                file_hash="c_empty",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "b_empty").first()
            color = s.query(Image).filter(Image.file_hash == "c_empty").first()
            svc = DeoldifiedPairingService(s)
            assert svc.sync_pair_data(orig, color) is None

    def test_copies_from_color_to_bw_when_bw_empty(self, tmp_db) -> None:
        """Direction follows the data: a filled colorized image fills the B&W."""
        from app.db.database import session_scope
        from app.db.models import Image, Person

        with session_scope() as s:
            orig = Image(
                file_path="/bw/r.jpg", file_hash="bw_empty2", file_mtime=0.0
            )
            color = Image(
                file_path="/color/r-deoldified.jpg",
                file_hash="color_full2",
                file_mtime=0.0,
            )
            s.add_all([orig, color])
            s.flush()
            p = Person(name="Béla", is_auto_named=False)
            s.add(p)
            s.flush()
            s.add(_make_face(color.id, p.id))

        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "bw_empty2").first()
            color = s.query(Image).filter(Image.file_hash == "color_full2").first()
            svc = DeoldifiedPairingService(s)
            result = svc.sync_pair_data(orig, color)
            assert result is not None
            assert result["source_id"] == color.id
            assert result["target_id"] == orig.id

        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "bw_empty2").first()
            assert len(orig.faces) == 1
            assert orig.faces[0].person.name == "Béla"

    def test_metadata_only_counts_as_data(self, tmp_db) -> None:
        """An image with only a note (no faces) still counts as having data."""
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            orig = Image(
                file_path="/bw/m.jpg",
                file_hash="meta_only",
                file_mtime=0.0,
                note="only a note",
            )
            color = Image(
                file_path="/color/m-deoldified.jpg",
                file_hash="meta_empty",
                file_mtime=0.0,
            )
            s.add_all([orig, color])

        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "meta_only").first()
            color = s.query(Image).filter(Image.file_hash == "meta_empty").first()
            svc = DeoldifiedPairingService(s)
            assert svc.image_has_data(orig) is True
            assert svc.image_has_data(color) is False
            result = svc.sync_pair_data(orig, color)
            assert result is not None
            assert result["faces_copied"] == 0
            assert result["metadata_fields"] == ["note"]


class TestExtractVariantLabel:
    def test_artistic(self) -> None:
        assert extract_variant_label("photo-deoldified (artistic)") == "(artistic)"

    def test_stable(self) -> None:
        assert extract_variant_label("photo-deoldified (stable)") == "(stable)"

    def test_plain_deoldified(self) -> None:
        assert extract_variant_label("photo-deoldified") == "deoldified"

    def test_not_deoldified(self) -> None:
        assert extract_variant_label("normal_photo") == ""

    def test_case_insensitive(self) -> None:
        assert extract_variant_label("photo-DEOLDIFIED (artistic)") == "(artistic)"


class TestFindAllDeoldifiedForOriginal:
    def test_returns_every_variant_sorted_by_label(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            s.add_all([
                Image(file_path="/f/photo.jpg", file_hash="orig", file_mtime=0.0),
                Image(file_path="/f/photo-deoldified (stable).jpg",
                      file_hash="stable", file_mtime=0.0),
                Image(file_path="/f/photo-deoldified (artistic).jpg",
                      file_hash="artistic", file_mtime=0.0),
                Image(file_path="/f/unrelated.jpg", file_hash="other", file_mtime=0.0),
            ])

        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "orig").first()
            svc = DeoldifiedPairingService(s)
            variants = svc.find_all_deoldified_for_original(orig)
            assert [v.file_hash for v in variants] == ["artistic", "stable"]
            # The single-pair wrapper still returns the first.
            assert svc.find_deoldified_for_original(orig).file_hash == "artistic"


class TestGetComparisonGroup:
    def _seed(self, session) -> None:
        from app.db.models import Image
        session.add_all([
            Image(file_path="/f/photo.jpg", file_hash="orig", file_mtime=0.0),
            Image(file_path="/f/photo-deoldified (artistic).jpg",
                  file_hash="artistic", file_mtime=0.0),
            Image(file_path="/f/photo-deoldified (stable).jpg",
                  file_hash="stable", file_mtime=0.0),
        ])

    def test_group_from_original_orders_bw_first(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            self._seed(s)
        with session_scope() as s:
            orig = s.query(Image).filter(Image.file_hash == "orig").first()
            group = DeoldifiedPairingService(s).get_comparison_group(orig)
            assert [m.is_bw for m in group] == [True, False, False]
            assert [m.label for m in group] == ["", "(artistic)", "(stable)"]
            assert all(isinstance(m, ComparisonMember) for m in group)

    def test_group_from_a_colorized_variant_is_identical(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            self._seed(s)
        with session_scope() as s:
            stable = s.query(Image).filter(Image.file_hash == "stable").first()
            group = DeoldifiedPairingService(s).get_comparison_group(stable)
            assert len(group) == 3
            assert group[0].is_bw is True
            assert [m.label for m in group[1:]] == ["(artistic)", "(stable)"]

    def test_no_group_for_lone_image(self, tmp_db) -> None:
        from app.db.database import session_scope
        from app.db.models import Image

        with session_scope() as s:
            s.add(Image(file_path="/f/solo.jpg", file_hash="solo", file_mtime=0.0))
        with session_scope() as s:
            solo = s.query(Image).filter(Image.file_hash == "solo").first()
            assert DeoldifiedPairingService(s).get_comparison_group(solo) == []
