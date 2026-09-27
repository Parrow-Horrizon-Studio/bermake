"""image_paths: the shared write-side sanitizer and the sibling-read size ceiling."""

from bermake.io.image_paths import _MAX_IMAGE_BYTES, read_sibling_image_bytes, sanitize_filename_stem

FAKE_PNG = b"\x89PNG\r\n\x1a\n-fake-bytes"


def test_whitespace_still_collapses_for_a_readable_stem():
    assert sanitize_filename_stem("Brick Red", "material") == "Brick_Red"
    assert sanitize_filename_stem("  a  b ", "material") == "a_b"


def test_empty_name_falls_back():
    assert sanitize_filename_stem("", "material") == "material"


def test_a_traversal_name_never_produces_a_path_separator():
    """Finding 1 (M7.5c-3 whole-branch review): a texture or material name is
    untrusted text (on import a texture is named after the source document's
    own material name), and it becomes a `Path.with_name` filename. This does
    not need to prevent traversal itself -- `Path.with_name` already rejects
    a separator outright -- but the result must never contain one, or a
    caller relying on that rejection would be relying on an accident."""
    stem = sanitize_filename_stem("../../pwned", "image")
    assert "/" not in stem
    assert "\\" not in stem


def test_a_drive_or_ads_colon_is_replaced():
    """'a:b' passes `Path.with_name` on Windows but names an NTFS alternate
    data stream, which is not the file the caller thinks it is writing."""
    assert ":" not in sanitize_filename_stem("a:b", "image")


def test_uri_significant_characters_are_replaced():
    """'#', '?', '%' are not rejected by `Path.with_name` but are significant
    in a URI, and glTF's `image.uri` requires them percent-encoded; passing
    them through unescaped writes a spec-invalid document."""
    for bad in ("a#b", "a?b", "a%b"):
        stem = sanitize_filename_stem(bad, "image")
        assert stem == "a_b", stem


def test_a_name_entirely_of_unsafe_characters_falls_back_rather_than_going_empty():
    assert sanitize_filename_stem("###", "image") == "image"
    assert sanitize_filename_stem("///", "image") == "image"


def test_leading_and_trailing_underscores_are_not_stripped_from_a_meaningful_name():
    """Re-review regression (M7.5c-3): an earlier version of this function
    ended with `.strip("_")`, which unconditionally removed a leading or
    trailing underscore even when the rest of the name was perfectly safe.
    That silently collapsed "_Brick" to the same sanitized stem as "Brick"
    -- issue #119's exact failure mode, two distinct names merging into one
    on write, so one material and its image vanish from the export -- and
    widened its trigger set from "a literal duplicate name" to "any name
    with a leading or trailing underscore". A leading/trailing underscore
    that survives sanitizing must now be left alone."""
    assert sanitize_filename_stem("_Custom", "image") == "_Custom"
    assert sanitize_filename_stem("_Brick", "image") == "_Brick"
    assert sanitize_filename_stem("__a__", "image") == "__a__"


def test_a_leading_underscore_name_differs_from_its_bare_form():
    """The direct #119 pin: two materials named "_Brick" and "Brick" must
    export as two distinct newmtl blocks / two distinct glTF materials, not
    merge into one because sanitizing discarded the leading underscore."""
    assert sanitize_filename_stem("_Brick", "image") != sanitize_filename_stem("Brick", "image")


def test_names_with_no_alphanumeric_content_fall_back():
    """"." and ".." are each meaningful path components on their own -- "."
    the same directory, ".." the parent -- and neither happens to contain an
    alphanumeric character, so the same "no alphanumeric survived" rule that
    catches "___" and "###" catches them too, without needing a special case
    for either."""
    assert sanitize_filename_stem("___", "image") == "image"
    assert sanitize_filename_stem("###", "image") == "image"
    assert sanitize_filename_stem(".", "image") == "image"
    assert sanitize_filename_stem("..", "image") == "image"


def test_only_the_documented_safe_characters_survive():
    stem = sanitize_filename_stem("Brick-Red_02.v3!@$%^&*()", "image")
    assert set(stem) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")


def test_a_sibling_image_within_the_size_ceiling_is_read(tmp_path):
    (tmp_path / "small.png").write_bytes(FAKE_PNG)
    assert read_sibling_image_bytes(tmp_path, "small.png") == FAKE_PNG


def test_a_sibling_image_over_the_size_ceiling_is_refused(tmp_path, monkeypatch):
    """A .gltf/.obj shipped in an archive beside a multi-gigabyte 'texture' is
    a plausible hostile shape; read_gltf_scene already guards the document
    itself and every declared element count for exactly this reason, and an
    unbounded sibling read would undo that budget for one file the guard
    never looks at. The ceiling is monkeypatched down rather than writing a
    real oversized file to keep the test fast."""
    monkeypatch.setattr("bermake.io.image_paths._MAX_IMAGE_BYTES", 4)
    big = tmp_path / "huge.png"
    big.write_bytes(b"x" * 5)
    assert read_sibling_image_bytes(tmp_path, "huge.png") is None


def test_a_sibling_image_exactly_at_the_ceiling_is_read(tmp_path, monkeypatch):
    monkeypatch.setattr("bermake.io.image_paths._MAX_IMAGE_BYTES", 5)
    exact = tmp_path / "exact.png"
    exact.write_bytes(b"x" * 5)
    assert read_sibling_image_bytes(tmp_path, "exact.png") == b"x" * 5


def test_the_default_ceiling_is_a_few_hundred_mib():
    """Pins the constant's order of magnitude so a future edit that
    accidentally drops a zero (or removes the check entirely by setting it
    absurdly high) is caught here rather than only in a code review."""
    assert 100 * 1024 * 1024 <= _MAX_IMAGE_BYTES <= 1024 * 1024 * 1024
