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


def _envelope(ids, fact_count=None, next_page_token=None, key="traces"):
    """A response envelope shaped like the real API's."""
    body = {key: [{"trace_id": i} for i in ids]}
    if fact_count is not None:
        body["fact_count"] = fact_count
    if next_page_token is not None:
        body["next_page_token"] = next_page_token
    return body


def _extract_traces(envelope):
    """Pull trace ids out of an envelope, list-container style."""
    if not isinstance(envelope, dict):
        return []
    return [item["trace_id"] for item in envelope.get("traces", [])]


class TestIterPages:

    def test_the_token_is_sent_as_next_page_token(self):
        """The GET routes read 'next_page_token'; only the POST /evals/report
        pager spells it 'page_token'. Sending the wrong name is not an error --
        it is ignored and page 1 is re-served, which is indistinguishable from
        the bug being fixed. A mock that ignored param names could not catch it,
        so this one serves page 2 ONLY for the correct name.
        """
        calls = []

        def fetch(params):
            calls.append(dict(params))
            if params.get("next_page_token") is None:
                return _envelope(["a", "b"], fact_count=4, next_page_token="tok-2")
            assert params["next_page_token"] == "tok-2"
            return _envelope(["c", "d"], fact_count=4)

        collected = OkahuSpanLoader._collect_paged(
            fetch, {"start_time": "x"}, 200, _extract_traces, "traces in 'wf'")

        assert collected == ["a", "b", "c", "d"]
        assert "page_token" not in calls[1], "the POST pager's name must not be used"

    def test_page_size_is_sent_on_every_request(self):
        calls = []

        def fetch(params):
            calls.append(dict(params))
            if len(calls) == 1:
                return _envelope(["a"], fact_count=2, next_page_token="tok-2")
            return _envelope(["b"], fact_count=2)

        OkahuSpanLoader._collect_paged(fetch, {}, 50, _extract_traces, "ctx")

        assert [c["page_size"] for c in calls] == [50, 50]

    def test_a_single_page_issues_exactly_one_request(self):
        calls = []

        def fetch(params):
            calls.append(dict(params))
            return _envelope(["a", "b"], fact_count=2)

        collected = OkahuSpanLoader._collect_paged(
            fetch, {}, 200, _extract_traces, "ctx")

        assert collected == ["a", "b"]
        assert len(calls) == 1
        assert "next_page_token" not in calls[0]

    def test_three_pages_concatenate_in_server_order(self):
        pages = [
            _envelope(["a", "b"], fact_count=5, next_page_token="t2"),
            _envelope(["c", "d"], fact_count=5, next_page_token="t3"),
            _envelope(["e"], fact_count=5),
        ]

        def fetch(params):
            return pages.pop(0)

        assert OkahuSpanLoader._collect_paged(
            fetch, {}, 2, _extract_traces, "ctx") == ["a", "b", "c", "d", "e"]

    def test_an_empty_payload_gives_an_empty_list(self):
        collected = OkahuSpanLoader._collect_paged(
            lambda params: _envelope([], fact_count=0), {}, 200,
            _extract_traces, "ctx")

        assert collected == []

    def test_a_bare_list_envelope_does_not_crash(self):
        """Some mocks -- and the API's older shapes -- return a bare list rather
        than a dict. Reading the token off it must not raise AttributeError."""
        collected = OkahuSpanLoader._collect_paged(
            lambda params: [{"trace_id": "a"}], {}, 200,
            lambda env: [i["trace_id"] for i in env] if isinstance(env, list) else [],
            "ctx")

        assert collected == ["a"]

    def test_a_repeated_token_stops_the_walk(self, caplog):
        """A server echoing a token would spin the loop forever. Discovery runs
        at pytest collection time, so a hang is worse than a wrong answer."""
        calls = []

        def fetch(params):
            calls.append(dict(params))
            return _envelope(["a"], fact_count=99, next_page_token="same-token")

        collected = OkahuSpanLoader._collect_paged(
            fetch, {}, 200, _extract_traces, "traces in 'wf'")

        assert len(calls) == 2, "page 1, one follow-up, then stop"
        assert collected == ["a", "a"]
        assert "repeated a page token" in caplog.text


class TestReconciliation:

    def test_a_short_result_warns_naming_both_numbers(self, caplog):
        """The defect being fixed was silent. Never return short without saying so."""
        collected = OkahuSpanLoader._collect_paged(
            lambda params: _envelope(["a", "b"], fact_count=309), {}, 200,
            _extract_traces, "traces in 'wf'")

        assert collected == ["a", "b"]
        assert "2 of 309" in caplog.text
        assert "traces in 'wf'" in caplog.text

    def test_a_complete_result_is_silent(self, caplog):
        OkahuSpanLoader._collect_paged(
            lambda params: _envelope(["a", "b"], fact_count=2), {}, 200,
            _extract_traces, "ctx")

        assert "incomplete" not in caplog.text

    def test_no_fact_count_means_no_warning(self, caplog):
        """An envelope without fact_count advertises nothing to reconcile against."""
        OkahuSpanLoader._collect_paged(
            lambda params: _envelope(["a"]), {}, 200, _extract_traces, "ctx")

        assert "incomplete" not in caplog.text
