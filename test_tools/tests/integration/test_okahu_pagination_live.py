"""Live proof that the enumerators walk past the server's first page.

Against stage, not prod. The assertions are self-consistency checks rather than
literals: stage keeps ingesting, so "== 133" would rot within days, while
"collected everything the server says exists" stays true forever.

Credentials and the stage endpoint come from tests/integration/__init__.py.

At the time of writing this window holds 133 traces and 322 inferences, both
past the server's default page of 100. Before the fix get_trace_ids returned
100 and get_fact_ids returned 0.

Backlog issue #242.
"""
import os

import pytest

from monocle_test_tools.okahu_span_loader import OkahuSpanLoader

APP = "monoclepytest_4cvu17"
START, END = "2026-06-01T00:00:00.000Z", "2026-09-03T23:59:59.000Z"
SERVER_DEFAULT_PAGE = 100

# The app below lives on stage. Running this against prod would either 404 or
# walk an unrelated tenant's data, so the endpoint is checked rather than assumed
# -- _get_api_base falls back to the prod base URL when nothing is configured.
_ON_STAGE = "stage" in OkahuSpanLoader._get_api_base()

pytestmark = [
    pytest.mark.skipif(not os.environ.get("OKAHU_API_KEY"),
                       reason="needs OKAHU_API_KEY"),
    pytest.mark.skipif(not _ON_STAGE,
                       reason=f"needs the stage endpoint; got "
                              f"{OkahuSpanLoader._get_api_base()}"),
]


def _advertised(path_suffix, params):
    """fact_count straight off page 1, before any paging."""
    envelope = OkahuSpanLoader._get_resource(
        OkahuSpanLoader._get_api_base(), path_suffix,
        OkahuSpanLoader._get_headers(),
        params={"start_time": START, "end_time": END, **params},
        context_msg="probe")
    return envelope.get("fact_count")


def test_trace_enumeration_collects_every_page():
    advertised = _advertised(f"{APP}/traces", {})
    assert advertised > SERVER_DEFAULT_PAGE, (
        "this window must span more than one page or the test proves nothing")

    ids = OkahuSpanLoader.get_trace_ids(APP, start_time=START, end_time=END)

    assert len(ids) == advertised, "collected everything the server advertises"
    assert len(ids) > SERVER_DEFAULT_PAGE, "the 100-row truncation is gone"
    assert len(set(ids)) == len(ids), (
        "duplicates mean the token re-served page 1 instead of advancing")


def test_fact_enumeration_collects_every_page():
    advertised = _advertised(
        f"{APP}/facts/inferences/ids",
        {"duration_fact": "inferences", "breakdown_filter": "inferences"})
    assert advertised > SERVER_DEFAULT_PAGE

    ids = OkahuSpanLoader.get_fact_ids(APP, "inferences",
                                       start_time=START, end_time=END)

    assert len(ids) == advertised
    assert len(ids) > SERVER_DEFAULT_PAGE
    assert len(set(ids)) == len(ids)


def test_an_explicit_page_size_reaches_the_same_total():
    """Page depth must not change the answer -- only how many round trips it
    takes. A smaller page forces more token follow-ups over the same window."""
    wide = OkahuSpanLoader.get_trace_ids(APP, start_time=START, end_time=END,
                                         page_size=1000)
    narrow = OkahuSpanLoader.get_trace_ids(APP, start_time=START, end_time=END,
                                           page_size=25)

    assert sorted(wide) == sorted(narrow)
