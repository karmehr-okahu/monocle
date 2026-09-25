import pytest
from pathlib import Path
from monocle_test_tools.pytest_plugin import monocle_trace_asserter
from test_common.adk_travel_agent import root_agent

# All four tests share this custom template (classifies user input as
# valid / ambiguous / invalid relative to the agent's inferred purpose).
TEMPLATE_PATH = str(Path(__file__).parent / "custom_templates" / "user_input_validity.json")
# Route 1: an evidence_scope supplied as a JSON file on disk (loaded the same way
# TEMPLATE_PATH is). The file holds the full scope; trim it to narrow evidence.
EVIDENCE_SCOPE_PATH = str(Path(__file__).parent / "evidence_scopes" / "full_scope.json")


@pytest.mark.asyncio
async def test_custom_template_traces(monocle_trace_asserter):
    """Custom user_input_validity template, traces fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk", "Book a flight from San Jose to Seattle for 27th Nov 2025.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="traces",
        expected="valid",
    )


@pytest.mark.asyncio
async def test_custom_template_inferences(monocle_trace_asserter):
    """Custom user_input_validity template, inferences fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk", "I would like to order a small black coffee and a muffin for pickup at 10am.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="inferences",
        expected="invalid",
    )


@pytest.mark.asyncio
async def test_custom_template_agentic_turns(monocle_trace_asserter):
    """Custom user_input_validity template, agentic_turns fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk", "Please book a flight from Seattle to Portland for 10th April 2026. Could you also order me a coffee and a muffin for pickup at 10am? I'm going to need it.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="agentic_turns",
        expected="ambiguous",
    )


@pytest.mark.asyncio
async def test_custom_template_agentic_sessions(monocle_trace_asserter):
    """Custom user_input_validity template, agentic_sessions fact — valid booking request."""
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk", "Book a flight from Dallas to Houston for 1st May 2026 and book a hotel for 2 nights.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="agentic_sessions",
        expected="valid",
    )


@pytest.mark.asyncio
async def test_custom_template_evidence_scope(monocle_trace_asserter):
    """Custom template + evidence_scope scaffold.

    The scope below lists EVERYTHING (all span types, roles, events), which
    reproduces the broad default while exercising the evidence_scope path. To
    test a lever, delete lines from `evidence_scope` (e.g. drop "input", drop the
    "tool" role, or narrow span_types) and re-run — deploy the backend first.
    Change `fact_name` to run against a different fact; keep `group_by` matching
    it (or delete `group_by` to avoid the soft mismatch warning).
    """
    evidence_scope = {
        "span_types": [
            "agentic.turn",
            "inference",
            "inference.framework",
            "agentic.invocation",
            "agentic.tool.invocation",
            "agentic.mcp.invocation",
            "embedding",
            "embedding.modelapi",
            "retrieval",
            "retrieval.embedding",
        ],
        "roles": ["system", "user", "assistant", "tool"],
        "events": ["input", "output"],
        "group_by": "traces",  # keep matching fact_name below, or delete this line
    }
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk", "Book a flight from San Jose to Seattle for 27th Nov 2025.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="traces",
        evidence_scope=evidence_scope,
        expected="valid",
    )


@pytest.mark.asyncio
async def test_custom_template_evidence_scope_from_file(monocle_trace_asserter):
    """Route 1: evidence_scope loaded from a JSON file via evidence_scope_path.

    Same behavior as test_custom_template_evidence_scope above, but the scope is
    read from EVIDENCE_SCOPE_PATH (either a bare scope object or the
    {"evidence_scope": {...}} request-body shape) instead of being spelled out
    inline. To test a lever, edit the file — drop "input", drop the "tool" role,
    or narrow span_types — and re-run; deploy the backend first.
    """
    await monocle_trace_asserter.run_agent_async(
        root_agent, "google_adk", "Book a flight from San Jose to Seattle for 27th Nov 2025.",
    )
    monocle_trace_asserter.with_evaluation("okahu").check_eval(
        template_path=TEMPLATE_PATH,
        fact_name="traces",
        evidence_scope_path=EVIDENCE_SCOPE_PATH,
        expected="valid",
    )


if __name__ == "__main__":
    pytest.main([__file__])
