import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

def append_metrics_history(history_path, endpoint, model_type, input_files, metrics_map, notes=None):
    record = {
        "run_id": str(uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "model_type": model_type,
        "input_files": input_files,
        "metrics": metrics_map,
        "notes": notes,
    }

    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    return record

def read_metrics_history(history_path, limit=50):
    path = Path(history_path)
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    return records[-limit:]