import json
import sqlite3

import pytest

from scripts.campaign_20261110 import contracts as c, goal_reuse_policy as policy, group_lane as lane
from scripts.campaign_20261110.tests.test_group_lane import candidate, claim


def legacy_allocator(root):
    f,review=candidate(root,0,goal=7)
    accepted=claim(root,f,review)
    path=root/"dataset/quotas.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE groups RENAME TO new_groups")
        db.execute("CREATE TABLE groups (id TEXT PRIMARY KEY, goal TEXT UNIQUE NOT NULL, review TEXT NOT NULL, record TEXT NOT NULL)")
        db.execute("INSERT INTO groups SELECT * FROM new_groups")
        db.execute("DROP TABLE new_groups")
        db.execute("DELETE FROM metadata WHERE key=?",(policy.POLICY_KEY,))
    return path,accepted


def test_explicit_migration_preserves_all_accepted_rows_and_allows_new_repeated_goal(tmp_path):
    path,accepted=legacy_allocator(tmp_path)
    export=tmp_path/"dataset/groups/g0_group.json"
    original=export.read_bytes()
    with sqlite3.connect(path) as db:
        before=policy._content(db)
        assert policy.has_unique_goal_index(db)
    with pytest.raises(c.ContractError,match="explicit goal-reuse"):
        lane._db(path)
    result=policy.migrate(root=tmp_path,state_path=path)
    assert result["status"] == "committed" and result["accepted_records_unchanged"]
    assert result["policy"]["owner_quote"] == "Don't care about repeats between train and eval when generating for the whole benchmark."
    with sqlite3.connect(path) as db:
        assert policy._content(db) == before and not policy.has_unique_goal_index(db)
        assert json.loads(db.execute("SELECT value FROM metadata WHERE key=?",(policy.POLICY_KEY,)).fetchone()[0]) == policy.POLICY
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO groups SELECT * FROM groups LIMIT 1")
    assert export.read_bytes() == original
    f,review=candidate(tmp_path,1,goal=7)
    new=claim(tmp_path,f,review)
    assert new["canonical_goal_sha256"] == accepted["canonical_goal_sha256"]
    assert sorted(new["accepted"].values()) == ["eval_standard","train"]
    assert policy.migrate(root=tmp_path,state_path=path) == result


def test_allocator_rejects_an_unknown_goal_reuse_policy(tmp_path):
    path=tmp_path/"state.sqlite";db=lane._db(path)
    with db:
        db.execute("UPDATE metadata SET value=? WHERE key=?",('{}',policy.POLICY_KEY))
    db.close()
    with pytest.raises(c.ContractError,match="another goal-reuse policy"):
        lane._db(path)
