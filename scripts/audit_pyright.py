#!/usr/bin/env python
"""
Runs pyright in strict mode and counts errors.
Outputs .metrics/pyright_<date>.json
"""
import re

from _utils import run, save_metrics, timestamp

# Run pyright and capture summary
out, err, code = run(["pyright", "--stats"])

# Parse text output from pyright --stats
error_count = 0
warning_count = 0
error_types = {}

# Look for the summary line (e.g., "42 errors, 13 warnings, 0 information")
for line in out.splitlines():
    summary_match = re.search(r"(\d+)\s+errors?,\s+(\d+)\s+warnings?", line)
    if summary_match:
        error_count = int(summary_match.group(1))
        warning_count = int(summary_match.group(2))
        break

# Parse individual error lines to count types
for line in out.splitlines():
    if " - error:" in line:
        # Extract error type (e.g., "Type of ... is unknown")
        match = re.search(r" - error: (.+?)(?:\s+\(.*?\))?\s*$", line)
        if match:
            error_msg = match.group(1)
            # Categorize by first few words
            error_type = " ".join(error_msg.split()[:3])
            error_types[error_type] = error_types.get(error_type, 0) + 1

metrics = {
    "timestamp": timestamp(),
    "tool": "pyright",
    "error_count": error_count,
    "warning_count": warning_count,
    "exit_code": code,
    "error_types": error_types,
}

# Make metrics available for aggregator
globals()["metrics"] = metrics

save_metrics(f"pyright_{timestamp()[0:10]}", metrics)
