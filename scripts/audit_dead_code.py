#!/usr/bin/env python
"""
Runs vulture on the repo and writes:
  .metrics/vulture_<date>.json
Payload keys:
  - num_items
  - total_confidence
  - by_file[{path: str, items: list}]
"""
from _utils import run, save_metrics, timestamp, BASE_DIR
import json
import pathlib
import re

# Run vulture on blackmarble package
out, err, _ = run(["vulture", str(BASE_DIR/"blackmarble")])

# Parse the output (vulture doesn't have a stable JSON output format)
items = []
for line in out.splitlines():
    if line.strip():
        # Parse vulture output format: "path:line: unused ..."
        match = re.match(r"^(.+?):(\d+): (.+)$", line)
        if match:
            path, line_num, msg = match.groups()
            items.append({
                "path": path,
                "line": int(line_num),
                "message": msg
            })

# Group by file
by_file = {}
for item in items:
    path = item["path"]
    if path not in by_file:
        by_file[path] = []
    by_file[path].append(item)

# Calculate total confidence (estimated based on item count)
total_confidence = len(items) * 80  # min_confidence setting

metrics = {
    "timestamp": timestamp(),
    "tool": "vulture",
    "num_items": len(items),
    "total_confidence": total_confidence,
    "by_file": by_file,
}

# Make metrics available for aggregator
globals()["metrics"] = metrics

save_metrics(f"vulture_{timestamp()[0:10]}", metrics)