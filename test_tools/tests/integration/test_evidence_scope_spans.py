"""Evidence-scope SPAN_TYPES-lever matrix on a FIXED trace (internal; deterministic).

Tests the SPAN_TYPES lever in isolation: roles and events are held at FULL; only
evidence_scope["span_types"] varies across every non-empty combination of these
FIVE span units (embedding / retrieval / http spans are intentionally excluded):

  1. agentic.turn
  2. inference               (== inference + inference.framework — always grouped;
                              they are the same span, differing only by framework)
  3. agentic.invocation
  4. agentic.tool.invocation
  5. agentic.mcp.invocation

5 units -> 2**5 - 1 = 31 non-empty combinations, x 3 facts = 93 sweep cases.

NOTE — this lever is where the SILENT gap lives. A combo can be a *valid* scope
yet contain none of the spans present at a given fact level (e.g. excluding
inference while evaluating the inferences fact). That starves the judge of
evidence -> result=[] -> a confusing "Unexpected response format" error rather
than a clear message. Those failures are EXPECTED until that gap is fixed (ticket
TBD); this file doubles as the repro. (Empty span_types, by contrast, is a loud
API 400 — see test_spans_none.)

group_by is intentionally OMITTED: monocle's "agentic_turns" maps to the Okahu
fact "agent_requests" before submit, so a group_by of "agentic_turns" would be an
invalid enum. Omitting it avoids the mismatch (group_by is optional in v1).

Requires OKAHU_API_KEY + OKAHU_EVALUATION_ENDPOINT pointed at the deployed
evidence_scope backend (#1488 / PR #1535). Example:

    OKAHU_API_KEY=... OKAHU_EVALUATION_ENDPOINT=<deployed-eval-api>/api \
      OKAHU_API_ENDPOINT=<api-endpoint> MONOCLE_EVAL_MATRIX=1 \
      pytest tests/integration/test_evidence_scope_spans.py -v -s
"""
import itertools
import pytest
from pathlib import Path
from monocle_test_tools.pytest_plugin import monocle_trace_asserter

TEMPLATE_PATH = str(Path(__file__).parent / "custom_templates" / "user_input_validity.json")

# Deterministic source: a fixed, already-ingested Okahu trace + its workflow.
# NOTE: this file uses a DIFFERENT trace/workflow than the events/roles files, so
# these are hardcoded (not env-driven) to keep the trace<->workflow pairing intact.
TRACE_ID = "36dc60bf0f46bbfcb38fcfff7ac7c684"
WORKFLOW_NAME = "lg_travel_agent"

# Any label the user_input_validity template can emit — a smoke assertion that
# the judge returned a sane classification. Tighten per-case once baselines exist.
VALID_LABELS = ["valid", "invalid", "ambiguous"]

# Eval aggregation levels to sweep. Session is added later (needs a session id).
FACTS = ["traces", "inferences", "agentic_turns"]

# Held at FULL for the span_types matrix.
ALL_ROLES = ["system", "user", "assistant", "tool"]
ALL_EVENTS = ["input", "output"]

# Span units. inference + inference.framework always travel together. The
# embedding / retrieval / http families are intentionally excluded.
SPAN_UNITS = [
    ("agentic.turn", ["agentic.turn"]),
    ("inference", ["inference", "inference.framework"]),
    ("agentic.invocation", ["agentic.invocation"]),
    ("agentic.tool.invocation", ["agentic.tool.invocation"]),
    ("agentic.mcp.invocation", ["agentic.mcp.invocation"]),
]


def _unit_subsets(units):
    """Every non-empty subset of the span UNITS, as (id, span_types) — the id is
    the '+'-joined unit names; span_types flattens each unit's span list (so the
    inference unit expands to both inference and inference.framework)."""
    out = []
    for r in range(1, len(units) + 1):
        for combo in itertools.combinations(units, r):
            combo_id = "+".join(u[0] for u in combo)
            span_types = [s for u in combo for s in u[1]]
            out.append((combo_id, span_types))
    return out


# All 31 non-empty span-unit combinations.
SPAN_COMBOS = _unit_subsets(SPAN_UNITS)


def _scope(span_types):
    # roles + events full; only span_types varies. group_by omitted (see docstring).
    return {"span_types": span_types, "roles": ALL_ROLES, "events": ALL_EVENTS}


@pytest.mark.parametrize("fact_name", FACTS)
@pytest.mark.parametrize("combo_id,span_types", SPAN_COMBOS, ids=[c[0] for c in SPAN_COMBOS])
@pytest.mark.asyncio
async def test_spans_combo(monocle_trace_asserter, combo_id, span_types, fact_name):
    """Sweep one span_types combination against one fact, on the fixed trace."""
    (monocle_trace_asserter
        .with_trace_source("okahu", id=TRACE_ID, fact_name="trace", workflow_name=WORKFLOW_NAME)
        .with_evaluation("okahu")
        .check_eval(
            template_path=TEMPLATE_PATH,
            fact_name=fact_name,
            evidence_scope=_scope(span_types),
            expected=VALID_LABELS,
        ))


@pytest.mark.parametrize("fact_name", FACTS)
@pytest.mark.asyncio
async def test_spans_none(monocle_trace_asserter, fact_name):
    """span_types=[] — hard-rejected by the API (LOUD 400, 'span_types' must be
    non-empty), surfaced as AssertionError. Goes RED; that red is the loud
    rejection working. Wrap in pytest.raises(AssertionError) for green.
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
