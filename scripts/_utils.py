import json
import pathlib
import subprocess
import re
import datetime

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
METRIC_DIR = BASE_DIR / ".metrics"
METRIC_DIR.mkdir(exist_ok=True)

def timestamp():
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def run(cmd: list[str]) -> tuple[str, str, int]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = proc.communicate()
    return out, err, proc.returncode

def save_metrics(name: str, payload: dict):
    outfile = METRIC_DIR / f"{name}.json"
    outfile.write_text(json.dumps(payload, indent=2))
    print(f"saved {outfile}")