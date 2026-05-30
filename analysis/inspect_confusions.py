# Extract selected true-vs-predicted domain confusions from cluster assignments.

from pathlib import Path
import json

path = Path("output/clustering/wavlm/cluster_assignments.jsonl")

rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        rows.append(row)

matches = [row for row in rows if row["domain"] == "gigaspeech" and row["predicted_domain"] == "librispeech"]

out_path = Path("analysis/wavlm/gigaspeech-librispeech-confusions.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for row in matches:
        f.write(json.dumps(row) + "\n")