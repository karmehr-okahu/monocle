import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
from monocle_test_tools.fluent_api import TraceAssertion


INNER_TEMPLATE = {
    "name": "test_template",
    "eval_prompt": "x",
    "structure_output": {
        "label": {"enums": ["a", "b"], "description": "x"},
        "explanation": {"description": "x"},
    },
}

# A bare evidence_scope object (route 2's inline shape).
SCOPE = {
    "span_types": ["agentic.turn", "inference"],
    "roles": ["user", "assistant"],
    "events": ["input", "output"],
}
# The request-body shape a file may also carry.
WRAPPED_SCOPE = {"evidence_scope": SCOPE}


def _make_asserter(eval_result=("a", "ok")):
    """Build a TraceAssertion with one span and a mocked evaluator."""
    span = MagicMock()
    eval_mock = MagicMock()
    eval_mock.evaluate.return_value = eval_result
    return TraceAssertion(filtered_spans=[span], _eval=eval_mock)


@pytest.fixture(autouse=True)
def _reset_trace_assertion_class_state():
    """Reset shared class-level TraceAssertion state before AND after each test.

    See the identically-named fixture in test_check_eval_template_path.py for the
    full rationale: several tests here build a TraceAssertion() directly and
    intentionally leave a recorded assertion behind, and pytest_plugin.py flips a
    passing test to failed when TraceAssertion().has_assertions() is true.
    """
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None
    yield
    TraceAssertion._assertion_errors = []
    TraceAssertion._eval_report = None
    TraceAssertion._okahu_filter = None


class TestCheckEvalEvidenceScopePath:

    def test_bare_scope_file_reaches_evaluate(self, tmp_path):
        """A file holding just the scope object is loaded and reaches evaluate()
        as evidence_scope, exactly as an inline evidence_scope dict would."""
        path = tmp_path / "scope.json"
        path.write_text(json.dumps(SCOPE), encoding="utf-8")
        asserter = _make_asserter(eval_result=("a", "ok"))

        result = asserter.check_eval(
            template=dict(INNER_TEMPLATE), expected="a",
            evidence_scope_path=str(path))

        assert not result.has_assertions(), result.get_assertion_messages()
        _, kwargs = asserter._eval.evaluate.call_args
        assert kwargs["evidence_scope"] == SCOPE

    def test_wrapped_scope_file_is_unwrapped_before_evaluate(self, tmp_path):
        """A file with {"evidence_scope": {...}} is unwrapped so evaluate() gets
        just the inner scope dict."""
        path = tmp_path / "scope.json"
        path.write_text(json.dumps(WRAPPED_SCOPE), encoding="utf-8")
        asserter = _make_asserter(eval_result=("a", "ok"))

        result = asserter.check_eval(
            template=dict(INNER_TEMPLATE), expected="a",
            evidence_scope_path=str(path))

        assert not result.has_assertions(), result.get_assertion_messages()
        _, kwargs = asserter._eval.evaluate.call_args
        assert kwargs["evidence_scope"] == SCOPE  # unwrapped

    def test_path_resolves_identically_to_inline_scope(self, tmp_path):
        """Route 1 (file) and route 2 (inline dict) send the same thing to
        evaluate() for the same scope."""
        path = tmp_path / "scope.json"
        path.write_text(json.dumps(SCOPE), encoding="utf-8")

        a_file = _make_asserter()
        a_file.check_eval(template=dict(INNER_TEMPLATE), expected="a",
                          evidence_scope_path=str(path))
        a_inline = _make_asserter()
        a_inline.check_eval(template=dict(INNER_TEMPLATE), expected="a",
                            evidence_scope=dict(SCOPE))

        assert (a_file._eval.evaluate.call_args.kwargs["evidence_scope"]
                == a_inline._eval.evaluate.call_args.kwargs["evidence_scope"])

    def test_missing_file_raises_clean_assertion(self, tmp_path):
        missing = tmp_path / "does_not_exist.json"
        asserter = _make_asserter()

        # See the try/finally comment in test_check_eval_template_path.py.
        try:
            result = asserter.check_eval(
                template=dict(INNER_TEMPLATE), expected="a",
                evidence_scope_path=str(missing))

            assert result.has_assertions()
            msg = result.get_assertion_messages()
            assert "Evidence scope file not found" in msg
            assert str(missing) in msg
        finally:
            TraceAssertion._assertion_errors = []

    def test_invalid_json_raises_clean_assertion(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json", encoding="utf-8")
        asserter = _make_asserter()

        try:
            result = asserter.check_eval(
                template=dict(INNER_TEMPLATE), expected="a",
                evidence_scope_path=str(path))

            assert result.has_assertions()
            msg = result.get_assertion_messages()
            assert "Evidence scope file is not valid JSON" in msg
            assert str(path) in msg
        finally:
            TraceAssertion._assertion_errors = []

    def test_non_object_json_raises_clean_assertion(self, tmp_path):
        """A JSON file that parses to a list (not an object) is rejected."""
        path = tmp_path / "list.json"
        path.write_text(json.dumps(["agentic.turn"]), encoding="utf-8")
        asserter = _make_asserter()

        try:
            result = asserter.check_eval(
                template=dict(INNER_TEMPLATE), expected="a",
                evidence_scope_path=str(path))

            assert result.has_assertions()
            assert "must contain a JSON object" in result.get_assertion_messages()
        finally:
            TraceAssertion._assertion_errors = []

    def test_both_inline_and_path_raises_value_error(self, tmp_path):
        """Supplying both evidence_scope and evidence_scope_path is rejected."""
        path = tmp_path / "scope.json"
        path.write_text(json.dumps(SCOPE), encoding="utf-8")
        asserter = _make_asserter()

        with pytest.raises(ValueError, match="not both"):
            asserter.check_eval(
                template=dict(INNER_TEMPLATE), expected="a",
                evidence_scope=dict(SCOPE), evidence_scope_path=str(path))


if __name__ == "__main__":
    pytest.main([__file__])
