"""Static check of the evaluator surface the lab code depends on, for a merge of ``lab-vec-eval`` with another branch (no
simulator; nothing is imported, the sources are parsed).

1. Patch targets (failures): what ``lab_tools/evaluator_patches.py`` replaces or reads must still have the shape the patches
   assume, or a patch silently does nothing - a renamed ``ExecutorConfig`` field just gets a new attribute, a planner class
   bound inside ``run_instance`` escapes ``NoPlanner``, a missing ``binary_success_ever`` key makes the never-held epilogue
   rule apply to every episode. Every import of the consumers below must still resolve.
2. Lab-owned files (failure when they differ from ``--base`` in the working tree): the vectorized modules come from
   ``lab-vec-eval`` only; a merge must not change them.
3. Drift (report): every function / method / class the vectorized modules and the evaluation-path lab tools take from
   ``ispatialgym`` (their imports and ``variant.attr`` uses, plus the patched / re-implemented methods), compared between
   ``--base`` (default ``origin/lab-vec-eval``) and the target (the working tree, or ``--target REF``) on the AST with
   docstrings stripped. A changed method that a vectorized class overrides (``VecISpatialGymTask``, ``SlotLoader``) or
   re-implements (``VecStreamExecutor``, ``VecInstanceLoader``, the patches' ``Monitors.check`` / ``Executor.epilogue``) is
   listed with its counterpart: it needs the same change there, or the single-env and the vectorized evaluator stop
   agreeing.

    python lab_tools/cluster/check_eval_surface.py [--base origin/lab-vec-eval] [--target REF] [--out sanity/eval_surface.json] [--diff] [--strict]

Exit code 1 when a patch target is broken, an import no longer resolves, a lab-owned file changed, or (``--strict``) an
overridden / re-implemented method changed. ``--target origin/<branch>`` previews a merge for the files lab-vec-eval does
not touch (the merge takes the other branch's version of those); lab-owned files absent from that ref are reported, not failed.
"""

from __future__ import annotations

import argparse
import ast
import copy
import difflib
import json
import os
import subprocess
import sys

WT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: modules whose ``ispatialgym`` imports define the surface: the vectorized evaluator and the lab's evaluation path
#: (instance-recovery tools such as reloc_recover.py / sample_chosen.py drive the sampler and are left out)
CONSUMERS = (
    "ispatialgym/oracle/vec_loader.py", "ispatialgym/oracle/vec_stream.py", "ispatialgym/tasks/vec_task.py",
    "ispatialgym/eval/clone_frames.py", "lab_tools/evaluator_patches.py", "lab_tools/eval_episodes.py",
    "lab_tools/replay_episodes.py", "lab_tools/replay_policy.py", "lab_tools/act_policy.py", "lab_tools/vec_eval.py",
    "lab_tools/eval_worker.py", "lab_tools/first_frames.py",
)
#: files only lab-vec-eval writes (and the one line of ispatialgym/tasks/__init__.py that exports its task)
LAB_OWNED = ("ispatialgym/oracle/vec_loader.py", "ispatialgym/oracle/vec_stream.py", "ispatialgym/tasks/vec_task.py",
             "ispatialgym/eval/clone_frames.py", "ispatialgym/tests/test_vec_stream.py")
LAB_OWNED_SYMBOLS = {("ispatialgym/tasks/__init__.py", "VecISpatialGymTask")}
#: re-implementations that are not subclasses: (module, qualified name) -> where the lab's counterpart lives
MIRRORS = {
    ("ispatialgym/oracle/executor.py", "Executor.run_stream"): "ispatialgym/oracle/vec_stream.py VecStreamExecutor.run",
    ("ispatialgym/oracle/executor.py", "Executor.epilogue"):
        "ispatialgym/oracle/vec_stream.py VecStreamExecutor._begin_epilogue / _end_window / _finalize; lab_tools/evaluator_patches.py epilogue wrapper",
    ("ispatialgym/oracle/executor.py", "Executor._noop_actions"): "ispatialgym/oracle/vec_stream.py VecStreamExecutor._noop",
    ("ispatialgym/oracle/qa.py", "Monitors.check"): "lab_tools/evaluator_patches.py check (replaces it during streams)",
    ("ispatialgym/oracle/run_isg.py", "InstanceLoader.build_env"): "ispatialgym/oracle/vec_loader.py VecInstanceLoader.build_env",
    ("ispatialgym/oracle/run_isg.py", "InstanceLoader.restore"): "ispatialgym/oracle/vec_loader.py VecInstanceLoader.restore_batch",
    ("ispatialgym/oracle/run_isg.py", "InstanceLoader.attach"): "ispatialgym/oracle/vec_loader.py VecInstanceLoader.restore_batch (world reuse)",
    ("ispatialgym/oracle/run_isg.py", "run_instance"): "lab_tools/vec_eval.py (the per-slot episode loop of the policy / replay modes)",
}
#: surface named explicitly (patched or read by name) on top of what the consumers' imports give
EXTRA = list(MIRRORS) + [
    ("ispatialgym/oracle/executor.py", "ExecutorConfig"), ("ispatialgym/oracle/execute.py", "_patch_ag_joint_type"),
    ("ispatialgym/oracle/behavior_lerobot.py", "robot_overrides"), ("ispatialgym/oracle/qa.py", "Monitors._check_collision"),
    ("ispatialgym/oracle/qa.py", "Monitors._check_timeout"), ("ispatialgym/oracle/run_isg.py", "parse_args"),
]

_CACHE: dict = {}


def read(ref, path):
    """``path`` at git ``ref`` (None: the working tree); None when it does not exist there."""
    key = (ref, path)
    if key not in _CACHE:
        if ref is None:
            p = os.path.join(WT, path)
            _CACHE[key] = open(p).read() if os.path.isfile(p) else None
        else:
            r = subprocess.run(["git", "-C", WT, "show", f"{ref}:{path}"], capture_output=True, text=True)
            _CACHE[key] = r.stdout if r.returncode == 0 else None
    return _CACHE[key]


def list_py(ref, prefix):
    if ref is None:
        out = []
        for root, dirs, files in os.walk(os.path.join(WT, prefix)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            out += [os.path.relpath(os.path.join(root, f), WT) for f in files if f.endswith(".py")]
        return sorted(out)
    r = subprocess.run(["git", "-C", WT, "ls-tree", "-r", "--name-only", ref, prefix], capture_output=True, text=True)
    return sorted(p for p in r.stdout.split() if p.endswith(".py"))


def module_file(mod, ref):
    base = mod.replace(".", "/")
    for cand in (base + ".py", base + "/__init__.py"):
        if read(ref, cand) is not None:
            return cand
    return None


def attr_chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id] + parts[::-1]
    return None


def surface_of(src, ref):
    """(module file, qualified name) pairs one consumer takes from ispatialgym, and its classes' bases among them."""
    tree = ast.parse(src)
    mods, syms, out = {}, {}, set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "ispatialgym":
            for a in node.names:
                local = a.asname or a.name
                sub = module_file(f"{node.module}.{a.name}", ref)
                if sub:
                    mods[local] = sub
                    continue
                mf = module_file(node.module, ref) or node.module.replace(".", "/") + ".py"
                syms[local] = (mf, a.name)
                out.add((mf, a.name))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] == "ispatialgym" and a.asname:
                    mf = module_file(a.name, ref)
                    if mf:
                        mods[a.asname] = mf
    bases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = attr_chain(node)
            if not chain or len(chain) < 2:
                continue
            if chain[0] in mods:
                out.add((mods[chain[0]], ".".join(chain[1:3])))
            elif chain[0] in syms:
                mf, name = syms[chain[0]]
                out.add((mf, f"{name}.{chain[1]}"))
        elif isinstance(node, ast.ClassDef):
            for b in node.bases:
                chain = attr_chain(b)
                if chain and chain[0] in syms and len(chain) == 1:
                    bases[node.name] = (syms[chain[0]], {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
                elif chain and chain[0] in mods and len(chain) == 2:
                    bases[node.name] = ((mods[chain[0]], chain[1]), {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
    return out, bases


def _defs_in(body, idx, prefix=""):
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            idx[prefix + node.name] = node
            if isinstance(node, ast.ClassDef) and not prefix:
                _defs_in(node.body, idx, prefix=node.name + ".")
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    idx.setdefault(prefix + t.id, node)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for e in t.elts:
                        if isinstance(e, ast.Name):
                            idx.setdefault(prefix + e.id, node)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            idx.setdefault(prefix + node.target.id, node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and not prefix:
            for a in node.names:
                idx.setdefault((a.asname or a.name).split(".")[0], node)
        elif isinstance(node, (ast.If, ast.Try)) and not prefix:
            _defs_in(node.body, idx)
            for h in getattr(node, "handlers", []):
                _defs_in(h.body, idx)
            _defs_in(node.orelse, idx)


def index(src):
    """{qualified name: node}: a module's top-level defs, classes, assignments and imported names, and each class's methods
    and class-level assignments as ``Class.name``."""
    idx = {}
    _defs_in(ast.parse(src).body, idx)
    return idx


def norm(node):
    """AST dump without docstrings; a class without its methods (compared one by one) but with their names."""
    node = copy.deepcopy(node)
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body and isinstance(n.body[0], ast.Expr) \
                and isinstance(getattr(n.body[0], "value", None), ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body = n.body[1:] or [ast.Pass()]
    if isinstance(node, ast.ClassDef):
        methods = sorted(b.name for b in node.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)))
        node.body = [b for b in node.body if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))] or [ast.Pass()]
        return ast.dump(node, include_attributes=False) + "|methods:" + ",".join(methods)
    return ast.dump(node, include_attributes=False)


def segment(src, node, limit=400):
    s = ast.get_source_segment(src, node) or ""
    if isinstance(node, ast.ClassDef):  # the class line and its non-method body only
        keep = [b for b in node.body if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))]
        s = "\n".join([s.splitlines()[0]] + [ast.get_source_segment(src, b) or "" for b in keep])
    lines = s.splitlines()
    return lines[:limit]


# ----------------------------------------------------------------------------------------------------------------- patch targets


def _func(idx, name):
    n = idx.get(name)
    return n if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def _argnames(fn):
    a = fn.args
    return [x.arg for x in a.posonlyargs + a.args], bool(a.vararg or a.kwarg or a.kwonlyargs)


def _bound_in(fn):
    names = {x.arg for x in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in n.names}
    return names


RUN_ISG, QA, EXECUTOR = "ispatialgym/oracle/run_isg.py", "ispatialgym/oracle/qa.py", "ispatialgym/oracle/executor.py"
EXECUTE, BL = "ispatialgym/oracle/execute.py", "ispatialgym/oracle/behavior_lerobot.py"


def check_patch_targets(ref) -> tuple[list, list]:
    """(file, message) failures and warnings of the patch targets in ``ref`` (None: the working tree)."""
    fails, warns = [], []

    def need(path):
        src = read(ref, path)
        if src is None:
            fails.append((path, "missing"))
        return src

    run_isg, qa, executor, execute, bl = need(RUN_ISG), need(QA), need(EXECUTOR), need(EXECUTE), need(BL)
    if run_isg:
        idx = index(run_isg)
        if "R1ProCuRobo" not in idx:
            fails.append((RUN_ISG, "no module-level R1ProCuRobo (NoPlanner replaces run_isg.R1ProCuRobo)"))
        ri = _func(idx, "run_instance")
        if ri is None:
            fails.append((RUN_ISG, "run_instance missing (eval_episodes / replay_episodes call it)"))
        elif "R1ProCuRobo" in _bound_in(ri):
            fails.append((RUN_ISG, "run_instance binds R1ProCuRobo locally: NoPlanner (a module attribute) would not apply"))
        elif not any(isinstance(n, ast.Name) and n.id == "R1ProCuRobo" for n in ast.walk(ri)):
            warns.append((RUN_ISG, "run_instance no longer constructs R1ProCuRobo: check how the planner is built now (NoPlanner may be bypassed)"))
        if _func(idx, "parse_args") is None:
            fails.append((RUN_ISG, "parse_args missing (the lab drivers build run_isg's arguments with it)"))
        for m in ("build_env", "attach", "restore"):
            if _func(idx, f"InstanceLoader.{m}") is None:
                fails.append((RUN_ISG, f"InstanceLoader.{m} missing (EvalLoader / SlotLoader / VecInstanceLoader rely on it)"))
        cls = idx.get("InstanceLoader")
        if isinstance(cls, ast.ClassDef):
            src = ast.get_source_segment(run_isg, cls) or ""
            for attr in ("behavior_output", "self.robot", "self.report"):
                if attr not in src:
                    fails.append((RUN_ISG, f"InstanceLoader has no {attr!r} (EvalLoader overrides behavior_output and writes report['fixed_handle_ag'] on self.robot)"))
    if qa:
        idx = index(qa)
        chk = _func(idx, "Monitors.check")
        if chk is None:
            fails.append((QA, "Monitors.check missing (replaced by the patches during streams)"))
        elif _argnames(chk) != (["self", "step"], False):
            fails.append((QA, f"Monitors.check signature {_argnames(chk)[0]}: the replacement takes (self, step=None)"))
        for m in ("_check_collision", "_check_timeout"):
            fn = _func(idx, f"Monitors.{m}")
            if fn is None:
                fails.append((QA, f"Monitors.{m} missing (the replacement calls it)"))
            elif _argnames(fn) != (["self"], False):
                fails.append((QA, f"Monitors.{m} signature {_argnames(fn)[0]}: the replacement calls it without arguments"))
        cls = idx.get("Monitors")
        src = (ast.get_source_segment(qa, cls) or "") if isinstance(cls, ast.ClassDef) else ""
        for attr in ("self.episode_steps", "self.action_steps"):
            if attr not in src:
                fails.append((QA, f"Monitors has no {attr} (the replacement keeps these counters)"))
    if executor:
        idx = index(executor)
        ep = _func(idx, "Executor.epilogue")
        if ep is None:
            fails.append((EXECUTOR, "Executor.epilogue missing (wrapped by the patches, mirrored by VecStreamExecutor)"))
        elif _argnames(ep) != (["self", "verdict", "n_action_steps"], False):
            fails.append((EXECUTOR, f"Executor.epilogue signature {_argnames(ep)[0]}: the wrapper takes (self, verdict, n_action_steps)"))
        cfg = idx.get("ExecutorConfig")
        fields = {b.target.id for b in getattr(cfg, "body", []) if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name)}
        for f in ("settle_min_steps", "settle_fraction", "settle_second_window_factor"):
            if f not in fields:
                fails.append((EXECUTOR, f"ExecutorConfig.{f} missing (the never-held epilogue rule sets it; a missing field becomes a no-op attribute)"))
        cls = idx.get("Executor")
        if isinstance(cls, ast.ClassDef) and "self.isg" not in (ast.get_source_segment(executor, cls) or ""):
            fails.append((EXECUTOR, "Executor has no self.isg (the epilogue wrapper reads binary_success_ever from it)"))
    if not [p for p in list_py(ref, "ispatialgym") if "/tests/" not in p and '"binary_success_ever"' in (read(ref, p) or "")]:
        fails.append(("ispatialgym/", "'binary_success_ever' is written nowhere: the never-held epilogue rule would apply to every episode"))
    if execute:
        fn = _func(index(execute), "_patch_ag_joint_type")
        if fn is None:
            fails.append((EXECUTE, "_patch_ag_joint_type missing (fixed_handle_ag, the default of every stream path)"))
        elif len(_argnames(fn)[0]) != 2:
            fails.append((EXECUTE, f"_patch_ag_joint_type signature {_argnames(fn)[0]}: the lab calls it as (robot, 'FixedJoint')"))
    if bl:
        fn = _func(index(bl), "robot_overrides")
        src = (ast.get_source_segment(bl, fn) or "") if fn else ""
        if fn is None:
            fails.append((BL, "robot_overrides missing (the no-depth patch wraps it)"))
        elif "obs_modalities" not in src:
            fails.append((BL, "robot_overrides has no 'obs_modalities' (the no-depth patch filters it)"))
        elif "depth_linear" not in bl:
            warns.append((BL, "no 'depth_linear' modality any more (the no-depth patch is a no-op)"))
    return fails, warns


# ----------------------------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/lab-vec-eval", help="git ref of the lab code the merge started from")
    ap.add_argument("--target", default=None, help="git ref to check instead of the working tree (merge preview)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--diff", action="store_true", help="print the diffs of the changed symbols")
    ap.add_argument("--strict", action="store_true", help="also fail when an overridden / re-implemented method changed")
    args = ap.parse_args(argv)
    base, target = args.base, args.target
    if subprocess.run(["git", "-C", WT, "rev-parse", "--verify", "-q", f"{base}^{{commit}}"], capture_output=True).returncode != 0:
        print(f"base ref {base!r} not found (git fetch origin lab-vec-eval?)", file=sys.stderr)
        return 2

    def rev(ref):
        r = subprocess.run(["git", "-C", WT, "rev-parse", ref or "HEAD"], capture_output=True, text=True)
        return r.stdout.strip() + ("" if ref else " + working tree")

    report = {"format": "isg-lab-eval-surface/v1", "base": [base, rev(base)], "target": [target or "working tree", rev(target)]}
    lab_edited = set()
    if target is not None:  # preview: files lab-vec-eval itself edits show the other branch's version, without the lab's edit
        mb = subprocess.run(["git", "-C", WT, "merge-base", base, target], capture_output=True, text=True).stdout.strip()
        if mb:
            r = subprocess.run(["git", "-C", WT, "diff", "--name-only", mb, base], capture_output=True, text=True)
            lab_edited = set(r.stdout.split())

    # 1. patch targets and the consumers' imports
    fails, warns = check_patch_targets(target)
    surface, overrides = set(EXTRA), {}
    consumers = {}
    for c in CONSUMERS:
        src = read(target, c) or read(base, c) or read(None, c)
        if src is None:
            warns.append((c, "consumer not found anywhere"))
            continue
        consumers[c] = "target" if read(target, c) is not None else ("base" if read(base, c) is not None else "working tree")
        s, b = surface_of(src, target if read(target, c) is not None else base)
        surface |= s
        for cls, ((mf, name), methods) in b.items():
            overrides.setdefault((mf, name), []).append((c, cls, methods))
    for mf, q in sorted(surface):
        head = q.split(".")[0]
        if (mf, head) in LAB_OWNED_SYMBOLS or mf in LAB_OWNED:
            continue
        src = read(target, mf)
        idx = index(src) if src is not None else {}
        if q in idx:
            continue
        owner = idx.get(head)
        if owner is None:
            fails.append((mf, f"{q} no longer resolves (imported / used by the lab code)"))
        elif isinstance(owner, ast.ClassDef) and "." in q:
            if owner.bases:  # may be inherited: reported, not failed
                warns.append((mf, f"{q} not defined in the class body (inherited from {[ast.unparse(b) for b in owner.bases]}?)"))
            else:
                fails.append((mf, f"{q} no longer resolves (imported / used by the lab code)"))
    unknown = [x for x in fails if x[0] in lab_edited]  # preview only: the merge would also carry the lab's side of the file
    fails = [x for x in fails if x[0] not in lab_edited]
    report["patch_targets"] = {"ok": not fails, "failures": [f"{p}: {m}" for p, m in fails], "warnings": [f"{p}: {m}" for p, m in warns],
                               "unknown_until_merged": [f"{p}: {m}" for p, m in unknown]}
    report["consumers"] = consumers

    # 2. lab-owned files
    owned = {}
    for p in LAB_OWNED:
        b, t = read(base, p), read(target, p)
        owned[p] = "identical" if b == t else ("absent in target" if t is None else "CHANGED")
    owned_fail = [p for p, s in owned.items() if s == "CHANGED" or (s == "absent in target" and target is None)]
    report["lab_owned"] = owned

    # 3. drift
    changed, unchanged = [], []
    expanded = set()
    for mf, q in surface:
        if mf in LAB_OWNED or (mf, q.split(".")[0]) in LAB_OWNED_SYMBOLS:
            continue
        expanded.add((mf, q))
        bsrc = read(base, mf)
        node = index(bsrc).get(q) if bsrc else None
        if isinstance(node, ast.ClassDef):  # every method of a class the lab subclasses or instantiates
            expanded |= {(mf, f"{q}.{b.name}") for b in node.body if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for mf, q in sorted(expanded):
        bsrc, tsrc = read(base, mf), read(target, mf)
        bn = index(bsrc).get(q) if bsrc else None
        tn = index(tsrc).get(q) if tsrc else None
        if bn is None and tn is None:
            continue
        item = {"module": mf, "name": q}
        if mf in lab_edited:
            item["lab_edited_file"] = True
        cls, _, meth = q.partition(".")
        if (mf, q) in MIRRORS:
            item["counterpart"] = MIRRORS[(mf, q)]
        if meth and (mf, cls) in overrides:
            ov = [f"{c} {k}.{meth}" for c, k, methods in overrides[(mf, cls)] if meth in methods]
            if ov:
                item["overridden_by"] = ov
        if bn is None or tn is None:
            item["status"] = "added in target" if bn is None else "REMOVED in target"
            changed.append(item)
            continue
        if norm(bn) == norm(tn):
            unchanged.append(f"{mf}: {q}")
            continue
        item["status"] = "CHANGED"
        item["diff"] = list(difflib.unified_diff(segment(bsrc, bn), segment(tsrc, tn), f"base:{mf}", f"target:{mf}", lineterm="", n=2))
        changed.append(item)
    critical = [c for c in changed if c.get("counterpart") or c.get("overridden_by")]
    report["drift"] = {"n_symbols": len(expanded), "unchanged": len(unchanged), "changed": changed, "needs_counterpart_review": [f"{c['module']}: {c['name']}" for c in critical]}

    ok = not fails and not owned_fail and not (args.strict and critical)
    report["pass"] = ok
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)

    print(f"base {report['base'][1][:12]} ({base})  target {report['target'][1][:12]} ({report['target'][0]})")
    print(f"patch targets: {'OK' if not fails else 'BROKEN'}")
    for line in report["patch_targets"]["failures"]:
        print("  FAIL", line)
    for line in report["patch_targets"]["warnings"]:
        print("  warn", line)
    for line in report["patch_targets"]["unknown_until_merged"]:
        print("  ?   ", line, "(lab-vec-eval edits this file too: decided by the real merge)")
    print("lab-owned files: " + ", ".join(f"{os.path.basename(p)} {s}" for p, s in owned.items()))
    print(f"drift: {len(expanded)} surface symbols, {len(unchanged)} unchanged, {len(changed)} changed")
    for c in changed:
        tag = ""
        if c.get("counterpart"):
            tag += f"  <- mirrored by {c['counterpart']}"
        if c.get("overridden_by"):
            tag += f"  <- overridden by {', '.join(c['overridden_by'])}"
        if c.get("lab_edited_file"):
            tag += "  (lab-vec-eval edits this file too: preview without the lab's edit)"
        print(f"  {c['status']:>17}  {c['module']}: {c['name']}{tag}")
        if args.diff and c.get("diff"):
            print("\n".join("      " + line for line in c["diff"][:200]))
    print("RESULT:", "PASS" if ok else "FAIL", "(--strict)" if args.strict else "")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
