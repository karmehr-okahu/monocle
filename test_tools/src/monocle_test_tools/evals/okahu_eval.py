import json
import logging
import os
from datetime import datetime, timezone
from opentelemetry.sdk.trace import Span
import requests
from monocle_apptrace.exporters.okahu import okahu_exporter
from monocle_apptrace.exporters.okahu.okahu_eval_result_exporter import OkahuEvalResultExporter
from monocle_test_tools.evals.base_eval import BaseEval
from typing import Optional, Union

logger = logging.getLogger(__name__)
OKAHU_PROD_EVALUATION_ENDPOINT = "https://eval.okahu.co/api"

class OkahuEval(BaseEval):
    def __init__(self, **data):
        eval_options = data.get("eval_options")
        super().__init__(eval_options=eval_options)
        self._exported_traces = set()  # Track which traces have been exported
    
    def export_trace(self, filtered_spans: list[Span]) -> str:
        """Export trace once and return trace_id"""
        if not filtered_spans:
            raise ValueError("No spans to export")
        
        span = filtered_spans[0]
        trace_id = format(span.get_span_context().trace_id, '032x')
        
        # Skip if already exported
        if trace_id in self._exported_traces:
            return trace_id
        
        # Export spans to Okahu
        exporter = okahu_exporter.OkahuSpanExporter(evaluate=True)
        exporter.export(filtered_spans)
        exporter.shutdown()
        
        self._exported_traces.add(trace_id)
        return trace_id

    def evaluate(self, filtered_spans: list[Span], eval_name: str, fact_name: str = "traces", eval_args: dict = {}) -> str:
        """Evaluate without exporting or deleting - just run the eval"""
        if not eval_name:
            raise ValueError("eval_name is required for evaluation.")
        
        api_key = (os.getenv("OKAHU_API_KEY")).strip()
        if not api_key:
            raise AssertionError("OKAHU_API_KEY is not configured.")
        
        # Get trace_id but DON'T export (assume already exported)
        span = filtered_spans[0]
        trace_id = format(span.get_span_context().trace_id, '032x')
        workflow_name = span.attributes.get("workflow.name")
        
        # Prepare eval job submission
        base = os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")
        submit_url = f"{base}/v1/eval/jobs"
        
        start_span_ns = span.start_time - 24 * 60 * 60 * 1e9
        end_span_ns = span.end_time + 24 * 60 * 60 * 1e9
        start = datetime.fromtimestamp(start_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        end = datetime.fromtimestamp(end_span_ns / 1e9, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        headers = {"x-api-key": api_key}
        payload = {"template_name": eval_name}
        params = {
            "workflow_name": workflow_name,
            "start_time": start,
            "end_time": end,
            "breakdown_filter": "traces",
            "trace_id": trace_id,
            "fact_name": "traces",
            "shadow_eval": True
        }
        
        # Submit evaluation job
        response = requests.post(url=submit_url, headers=headers, json=payload, params=params)
        response.raise_for_status()
        data = response.json()
        
        job_id = data.get("job_id")
        eval_result = data.get("result")
        label = json.loads(eval_result[0].get('result')).get('label')
        
        # Export results (but NOT delete trace here)
        if "okahu" in (os.getenv("MONOCLE_EXPORTER", "")):
            with OkahuEvalResultExporter(api_key=api_key, base_url=base) as result_exporter:
                result_exporter.export_results(
                    job_id=job_id,
                    eval_result=eval_result,
                    template_name=eval_name
                )
        
        return label
    
    def delete_trace(self, trace_id: str):
        """Delete trace from evaluation storage"""
        api_key = (os.getenv("OKAHU_API_KEY")).strip()
        base = os.getenv("OKAHU_EVALUATION_ENDPOINT", OKAHU_PROD_EVALUATION_ENDPOINT).rstrip("/")
        
        with OkahuEvalResultExporter(api_key=api_key, base_url=base) as result_exporter:
            result_exporter.delete_trace(trace_id=trace_id)
        
        self._exported_traces.discard(trace_id)