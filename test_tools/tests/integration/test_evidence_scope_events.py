"""Evidence-scope EVENTS-lever matrix on a FIXED trace (internal; deterministic).

Tests the EVENTS lever in isolation: roles are held at FULL (all four) and
span_types at FULL; only evidence_scope["events"] varies across every non-empty
combination. Sweeps the traces / inferences / agent_requests facts on one fixed,
already-ingested Okahu trace, so the evaluated content is deterministic.

group_by is intentionally OMITTED: monocle's "agentic_turns" maps to the Okahu
fact "agent_requests" before submit, so a group_by of "agentic_turns" would be an
invalid enum. Omitting it avoids the mismatch (group_by is optional in v1).

Requires OKAHU_API_KEY + OKAHU_EVALUATION_ENDPOINT pointed at the deployed
evidence_scope backend (#1488 / PR #1535). Example:

    OKAHU_API_KEY=... OKAHU_EVALUATION_ENDPOINT=<deployed-eval-api>/api \
      OKAHU_API_ENDPOINT=<api-endpoint> MONOCLE_EVAL_MATRIX=1 \
      pytest tests/integration/test_evidence_scope_events.py -v -s
"""
import itertools
import os
import pytest
from pathlib import Path
from monocle_test_tools.pytest_plugin import monocle_trace_asserter

TEMPLATE_PATH = str(Path(__file__).parent / "custom_templates" / "user_input_validity.json")

# Deterministic source: a fixed, already-ingested Okahu trace + its workflow.
TRACE_ID = "23b393d8c8ad775ea9fafaa960dbf165"
WORKFLOW_NAME = os.getenv("SCOPE_TEST_WORKFLOW", "karmehr-okahu-monocle")

# Any label the user_input_validity template can emit — a smoke assertion that
# the judge returned a sane classification. Tighten per-case once baselines exist.
VALID_LABELS = ["valid", "invalid", "ambiguous"]

# Eval aggregation levels to sweep. Session is added later (needs a session id).
FACTS = ["traces", "inferences", "agentic_turns"]

# Held at FULL for the events matrix.
ALL_ROLES = ["system", "user", "assistant", "tool"]
SPAN_TYPES = [
    "agentic.turn", "inference", "inference.framework",
    "agentic.invocation", "agentic.tool.invocation", "agentic.mcp.invocation",
    "embedding", "embedding.modelapi", "retrieval", "retrieval.embedding",
]
ALL_EVENTS = ["input", "output"]


def _subsets(items):
    """Every non-empty subset of items, as (id, list) — singles, then pairs, …"""
    out = []
    for r in range(1, len(items) + 1):
        for combo in itertools.combinations(items, r):
            out.append(("+".join(combo), list(combo)))
    return out


# All non-empty event combinations: {input}, {output}, {input+output}.
EVENT_COMBOS = _subsets(ALL_EVENTS)


def _scope(events):
    # roles + span_types full; only events varies. group_by omitted (see docstring).
    return {"span_types": SPAN_TYPES, "roles": ALL_ROLES, "events": events}


@pytest.mark.parametrize("fact_name", FACTS)
@pytest.mark.parametrize("combo_id,events", EVENT_COMBOS, ids=[c[0] for c in EVENT_COMBOS])
@pytest.mark.asyncio
async def test_events_combo(monocle_trace_asserter, combo_id, events, fact_name):
    """Sweep one events combination against one fact, on the fixed trace."""
    (monocle_trace_asserter
        .with_trace_source("okahu", id=TRACE_ID, fact_name="trace", workflow_name=WORKFLOW_NAME)
        .with_evaluation("okahu")
        .check_eval(
            template_path=TEMPLATE_PATH,
            fact_name=fact_name,
            evidence_scope=_scope(events),
            expected=VALID_LABELS,
        ))


@pytest.mark.parametrize("fact_name", FACTS)
@pytest.mark.asyncio
async def test_events_none(monocle_trace_asserter, fact_name):
    """events=[] — hard-rejected by the API (LOUD), surfaces as a test failure.

    The eval service validates evidence_scope and returns HTTP 400
    ("'events' must be a non-empty list"); monocle reformats that 400 as
    AssertionError("Custom template validation failed: ..."). So this test goes
    RED — and that red IS the loud rejection working, API-side (not monocle).
    Wrap the check_eval in pytest.raises(AssertionError) to keep the suite green
    while still asserting the rejection.

    NOTE: empty lists are NOT the silent case. The silent gap is a *valid* scope
    whose span_types are absent from the trace -> result=[] (tracked in a ticket).
    """
    (monocle_trace_asserter
        .with_trace_source("okahu", id=TRACE_ID, fact_name="trace", workflow_name=WORKFLOW_NAME)
        .with_evaluation("okahu")
        .check_eval(
            template_path=TEMPLATE_PATH,
            fact_name=fact_name,
            evidence_scope=_scope([]),
            expected=VALID_LABELS,
        ))


if __name__ == "__main__":
    pytest.main([__file__])
