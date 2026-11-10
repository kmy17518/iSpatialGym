"""Owner-authorized cross-group goal reuse and an audited allocator migration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import time

from scripts.campaign_20261110.contracts import canonical_json, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.storage import atomic_directory_lock, local_sqlite_transaction

POLICY = {
    "schema": "isg-goal-reuse-policy-v2", "version": 2,
    "owner_decision_date": "2026-11-10",
    "owner_quote": "Don't care about repeats between train and eval when generating for the whole benchmark.",
    "interpretation": "Repeated canonical goal states are permitted across independently generated groups and splits throughout the benchmark; no viewpoint variation is required to distinguish groups.",
    "cross_group_canonical_goal_unique": False,
    "fresh_generation_required": True, "group_id_unique": True,
    "within_group_distinct_starts_or_grasps": True, "max_group_members": 3,
    "standard_eval_requires_successful_training_sibling": True,
    "strict_model_category_room_holdouts_unchanged": True,
}
POLICY_KEY = "goal_reuse_policy"


def has_unique_goal_index(db):
    for row in db.execute("PRAGMA index_list(groups)"):
        if row[2]:
            name = row[1].replace('"','""')
            if [col[2] for col in db.execute(f'PRAGMA index_info("{name}")')] == ["goal"]:
                return True
    return False


def require_database_policy(db):
    require(not has_unique_goal_index(db), "explicit goal-reuse v2 policy migration required before new claims")
    previous = db.execute("SELECT value FROM metadata WHERE key=?",(POLICY_KEY,)).fetchone()
    require(previous is None or previous[0] == canonical_json(POLICY), "allocation database has another goal-reuse policy")
    if previous is None:
        db.execute("INSERT INTO metadata VALUES (?,?)",(POLICY_KEY,canonical_json(POLICY)))


def _content(db):
    result = {}
    for table in ("members","groups","assets","metadata"):
        rows = list(db.execute("SELECT * FROM "+table))
        if table == "metadata":
            rows = [row for row in rows if row[0] != POLICY_KEY]
        result[table] = {"rows":len(rows),"sha256":digest(sorted(rows))}
    return result


def migrate(*, root, state_path=None):
    """Remove only goal uniqueness; preserve accepted rows and group ownership."""
    root = Path(root).resolve()
    state = Path(state_path or root/"dataset/allocations.sqlite").resolve()
    require(state.is_relative_to(root) and state.is_file(), "missing owned allocation database")
    journal = root/"manifests/policy_migrations/goal_reuse_v2.json"
    with atomic_directory_lock(state.with_suffix(".lockdir")):
        prior = json.loads(journal.read_text()) if journal.exists() else None
        if prior is not None:
            require(prior["policy"] == POLICY and prior["database"] == str(state), "goal-reuse migration identity changed")
        with local_sqlite_transaction(state) as local:
            db = sqlite3.connect(local)
            try:
                before = _content(db)
                if prior is not None and prior["status"] == "committed":
                    require_database_policy(db)
                    db.commit()
                    return prior
                if prior is None:
                    prior = {"schema":"isg-goal-reuse-migration-v1","status":"prepared","database":str(state),
                        "created_at":time.time(),"before_database_sha256":file_digest(state),"before_content":before,
                        "previous_policy":{"version":1,"cross_group_canonical_goal_unique":True},"policy":POLICY}
                    write_json_atomic(journal,prior)
                require(before == prior["before_content"], "accepted data changed during unfinished goal-reuse migration")
                with db:
                    if has_unique_goal_index(db):
                        require([row[1] for row in db.execute("PRAGMA table_info(groups)")] == ["id","goal","review","record"],
                                "unexpected legacy group table schema")
                        require(not db.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='groups'").fetchall(),
                                "group migration requires explicit review of custom triggers")
                        db.execute("CREATE TABLE groups_goal_reuse_v2 (id TEXT PRIMARY KEY, goal TEXT NOT NULL, review TEXT NOT NULL, record TEXT NOT NULL)")
                        db.execute("INSERT INTO groups_goal_reuse_v2 SELECT id,goal,review,record FROM groups")
                        db.execute("DROP TABLE groups")
                        db.execute("ALTER TABLE groups_goal_reuse_v2 RENAME TO groups")
                    require_database_policy(db)
                    require(_content(db) == before, "goal-reuse migration changed accepted records")
                    require(db.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "goal-reuse migration damaged allocator")
                    require(db.execute("PRAGMA foreign_key_check").fetchall() == [], "goal-reuse migration changed foreign-key integrity")
            finally:
                db.close()
        result = {**prior,"status":"committed","committed_at":time.time(),"after_database_sha256":file_digest(state),
                  "accepted_records_unchanged":True,"group_id_uniqueness_preserved":True}
        write_json_atomic(journal,result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True);parser.add_argument("--state")
    args=parser.parse_args()
    result=migrate(root=args.root,state_path=args.state)
    print(json.dumps({"status":result["status"],"policy_version":POLICY["version"],"database":result["database"]}))


if __name__ == "__main__":
    main()
