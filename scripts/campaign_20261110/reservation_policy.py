"""Shared fail-closed guard for the campaign's active reservation revision."""
from pathlib import Path
import json

from scripts.campaign_20261110.contracts import digest, file_digest, require

POLICY_KEYS = ("model_holdout_models", "category_holdout_models", "category_holdout_categories",
               "category_holdout_rooms", "train_exclude_models", "pending_custom_hard_reservations_not_ready")


def policy_digest(reservations):
    return digest({key:reservations[key] for key in POLICY_KEYS})


def active_policy(root):
    root=Path(root).resolve()
    require(not (root/"dataset/reservation_migration_pending.json").exists(),
            "reservation policy migration is pending; claims must wait for its durable commit")
    path=root/"manifests/active_reservations.json"
    if not path.exists():
        return None
    pointer=json.loads(path.read_text())
    require(pointer.get("schema")=="isg-active-reservations-v1", "invalid active reservation pointer")
    data=[]
    for name in ("reservations","asset_hashes"):
        source=Path(pointer[name]["path"]).resolve()
        require(source.is_relative_to(root) and file_digest(source)==pointer[name]["sha256"], "active reservation source changed")
        data.append(json.loads(source.read_text()))
    require(policy_digest(data[0])==pointer["policy_sha256"], "active reservation policy digest changed")
    require(data[1]["reservation_sha256"]==pointer["reservations"]["sha256"], "active reservation byte audit differs")
    return pointer,data[0],data[1]


def require_active_policy(root,reservations):
    active=active_policy(root)
    if active is not None:
        require(policy_digest(reservations)==active[0]["policy_sha256"], "claim uses a superseded reservation policy")
