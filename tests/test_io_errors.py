from bermake.io.errors import BermakeFormatError, BermakeIOError, BermakeVersionError


def test_subclass_hierarchy():
    assert issubclass(BermakeFormatError, BermakeIOError)
    assert issubclass(BermakeVersionError, BermakeIOError)


def test_os_error_is_not_a_bermake_io_error():
    """BermakeIOError is a project-specific hierarchy; plain filesystem errors
    (missing file, permission denied) must NOT be silently caught by a
    `except BermakeIOError` handler — callers need OSError to still surface."""
    assert not issubclass(OSError, BermakeIOError)


def test_raisable_with_message():
    for exc in (BermakeIOError, BermakeFormatError, BermakeVersionError):
        try:
            raise exc("boom")
        except BermakeIOError as e:
            assert "boom" in str(e)
