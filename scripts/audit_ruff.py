#!/usr/bin/env python
"""
Runs ruff linter and counts issues.
Outputs .metrics/ruff_<date>.json
"""
import json
import re

from _utils import run, save_metrics, timestamp

out, _, _ = run(["ruff", "check", "blackmarble", "--output-format", "json"])

# Parse JSON output if available, otherwise parse text
try:
    issues = json.loads(out) if out else []
except json.JSONDecodeError:
    # Fallback to text parsing
    issues = []
    for line in out.splitlines():
        if re.match(r"^.*:\d+:\d+:", line):
            issues.append({"line": line})

# Count by rule code
rule_counts = {}
for issue in issues:
    if isinstance(issue, dict) and "code" in issue:
        code = issue["code"]
        rule_counts[code] = rule_counts.get(code, 0) + 1

metrics = {
    "timestamp": timestamp(),
    "tool": "ruff",
    "issue_count": len(issues),
    "rule_counts": rule_counts,
}

# Make metrics available for aggregator
globals()["metrics"] = metrics

save_metrics(f"ruff_{timestamp()[0:10]}", metrics)
