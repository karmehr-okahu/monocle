"""Pagination for the Okahu fact and trace enumerators.

The API pages with page_size out and next_page_token back. Monocle previously
sent neither, so every enumeration stopped at the server's DEFAULT_PAGE_SIZE of
100 with no log line, no tally and no exception -- backlog issue #242.
"""
import pytest

from monocle_test_tools.okahu_span_loader import OkahuSpanLoader


@pytest.fixture(autouse=True)
def _okahu_env(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "test-key")
    monkeypatch.setenv("OKAHU_API_ENDPOINT", "https://api.example")
    monkeypatch.delenv("OKAHU_API_TIMEOUT", raising=False)


class TestResolvePageSize:
    """Bounds are checked here so an out-of-range value never reaches the wire.

    The server answers a bad page_size with HTTP 400 rather than clamping, and
    that 400 surfaces mid-collection without naming the parameter that caused it.
    """

    def test_default_is_200(self):
        assert OkahuSpanLoader._resolve_page_size(None) == 200

    def test_an_explicit_size_is_honoured(self):
        assert OkahuSpanLoader._resolve_page_size(50) == 50

    def test_the_server_maximum_is_allowed(self):
        assert OkahuSpanLoader._resolve_page_size(1000) == 1000

    @pytest.mark.parametrize("bad", [0, -1, 1001])
    def test_out_of_range_raises_naming_the_bound(self, bad):
        with pytest.raises(ValueError, match="between 1 and 1000"):
            OkahuSpanLoader._resolve_page_size(bad)

    def test_a_string_raises(self):
        with pytest.raises(ValueError, match="must be an int"):
            OkahuSpanLoader._resolve_page_size("200")

    def test_a_bool_raises(self):
        """isinstance(True, int) is True in Python, so page_size=True would
        otherwise resolve to a page size of 1 rather than being rejected."""
        with pytest.raises(ValueError, match="must be an int"):
            OkahuSpanLoader._resolve_page_size(True)
