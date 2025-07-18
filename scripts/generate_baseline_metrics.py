#!/usr/bin/env python
"""
Runs all audit_* scripts and rolls them into a single baseline snapshot.
Creates .metrics/baseline_<YYYY-MM-DD>.json
"""
import json
import datetime
import pathlib
import subprocess
import sys
from _utils import METRIC_DIR, timestamp

# Add scripts dir to path
scripts_dir = pathlib.Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

def run_audit_script(script_name: str) -> dict:
    """Run an audit script and return its metrics."""
    script_path = scripts_dir / f"{script_name}.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True
    )
    
    # Read the generated metrics file
    # The audit scripts save files as toolname_YYYY-MM-DD.json
    tool_name = script_name.replace("audit_", "").replace("_", "-")
    if tool_name == "dead-code":
        tool_name = "vulture"
    elif tool_name == "pyright":
        tool_name = "pyright"
    elif tool_name == "ruff":
        tool_name = "ruff"
    
    # Find the most recent metrics file for this tool
    pattern = f"{tool_name}_*.json"
    metrics_files = list(METRIC_DIR.glob(pattern))
    if metrics_files:
        latest = max(metrics_files, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)
    
    return {}

# Run all audit scripts
# Note: pyright can be slow on first run, so we'll skip it for initial baseline
scripts = ["audit_dead_code", "audit_ruff"]  # TODO: Add "audit_pyright" once cached
snapshot = {"timestamp": timestamp(), "tools": {}}

for script in scripts:
    print(f"Running {script}...")
    metrics = run_audit_script(script)
    if metrics and "tool" in metrics:
        snapshot["tools"][metrics["tool"]] = metrics

# Add summary statistics
snapshot["summary"] = {
    "total_issues": sum(
        m.get("num_items", 0) + m.get("error_count", 0) + m.get("issue_count", 0)
        for m in snapshot["tools"].values()
    ),
    "tools_run": len(snapshot["tools"]),
}

# Save the baseline
today = datetime.date.today().isoformat()
path = METRIC_DIR / f"baseline_{today}.json"
path.write_text(json.dumps(snapshot, indent=2))
print(f"\nBaseline snapshot written to {path}")
print(f"Total issues found: {snapshot['summary']['total_issues']}")