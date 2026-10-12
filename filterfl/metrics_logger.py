"""Per-round CSV metrics logging."""
import csv
import os


class MetricsLogger:
    FIELDS = [
        "epoch", "ma", "asr", "active_rho", "is_detection_round",
        "sampled_clients", "adversarial_sampled", "detected_count",
        "detection_tp", "detection_fp",
        "per_client_confidence",  # json-ish string: client_id:pred:conf;...
        "per_client_trust",       # json-ish string: client_id:trust;... (siftfl only)
        "malicious_local_ma",     # json-ish string: client_id:ma;... (malicious sampled clients only)
    ]

    def __init__(self, path: str, append: bool = False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        write_header = not (append and os.path.exists(path))
        self._f = open(path, "a" if append else "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        if write_header:
            self._w.writeheader()

    def log(self, **kwargs):
        row = {k: kwargs.get(k, "") for k in self.FIELDS}
        self._w.writerow(row)
        self._f.flush()

    def close(self):
        self._f.close()
