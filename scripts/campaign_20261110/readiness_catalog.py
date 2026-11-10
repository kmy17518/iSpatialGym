"""Read the same readiness evidence as sampling.knowledge, without simulator imports."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


class ReadinessCatalog:
    def __init__(self, generated_data):
        self.root = Path(generated_data)
        self.complaints = defaultdict(list)
        paths = [self.root / n for n in ("complaints.json", "deletion_queue.csv", "object_inventory.json")]
        # Missing evidence is an error, not an implicit ready decision.
        for path in paths:
            if not path.is_file():
                raise ValueError(f"Missing authoritative readiness source {path}")
        for complaint in json.loads(paths[0].read_text()):
            if str(complaint.get("processed")).lower() != "true":
                self.complaints[complaint["object"].split("-")[-1]].append(complaint["type"])
        self.deleted = {row["Object"].strip().split("-")[-1] for row in csv.DictReader(paths[1].open()) if row.get("Object")}
        self.providers = set(json.loads(paths[2].read_text())["providers"])
        self.provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.source = json.dumps({"rule": "ispatialgym.sampling.knowledge.is_ready + provider presence",
                                  "files_sha256": self.provenance}, sort_keys=True, separators=(",", ":"))

    def status(self, category, model):
        known = f"{category}-{model}" in self.providers
        complaints = sorted(set(self.complaints.get(model, ())))
        deleted = model in self.deleted
        return {"ready": known and not complaints and not deleted, "readiness_source": self.source,
                "readiness_details": {"catalog_provider_present": known, "unprocessed_complaints": complaints,
                                      "queued_for_deletion": deleted}}
