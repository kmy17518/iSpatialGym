"""Join prepared articulation metadata to one private native customs entry.

No sampling, world construction, scheduling, or preparation registration is
performed. Scene/model construction must already match the selected row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ispatialgym.sampling.prepared_articulation import bind_generation, checked
from scripts.campaign_20261110.contracts import file_digest, require


def materialize(*, customs_ref, metadata_ref, selection_ref, configuration_id, variant, output):
    customs = checked(customs_ref)
    require(variant in customs, "exact generation variant missing from customs")
    customs[variant] = bind_generation(customs[variant], metadata_ref=metadata_ref,
        selection_ref=selection_ref, configuration_id=configuration_id, variant=variant)
    output = Path(output)
    require(not output.exists(), "private generation customs output already exists")
    with output.open("x") as stream:
        # Native grounding consumes role order; retain the source declaration's
        # insertion order when adding exact target options.
        stream.write(json.dumps(customs, indent=2, allow_nan=False) + "\n")
    return {"path": str(output.resolve()), "sha256": file_digest(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("customs", "metadata", "selection"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    for name in ("configuration-id", "variant", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        customs_ref={"path": args.customs, "sha256": args.customs_sha256},
        metadata_ref={"path": args.metadata, "sha256": args.metadata_sha256},
        selection_ref={"path": args.selection, "sha256": args.selection_sha256},
        configuration_id=args.configuration_id, variant=args.variant, output=args.output)))
