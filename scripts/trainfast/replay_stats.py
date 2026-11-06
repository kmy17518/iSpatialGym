import glob, json
F = "/vision/u/minyeong/projects/isg_home/trainfast"
for f in sorted(glob.glob(f"{F}/out/replay_test/*/replay_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    dem = [r for r in rows if r.get("tag") == "REPLAY_DEMO"]
    s = [r for r in rows if r.get("tag") == "REPLAY_SUMMARY"]
    name = "/".join(f.split("/")[-2:])
    fr = sum(r.get("frames", 0) for r in dem); w = sum(r.get("wall_s", 0) for r in dem)
    if s:
        s = s[-1]
        print(f"{name}: DONE demos={s['demos']} steps={s['env_steps']} steps/s={s['env_steps_per_s']} load_s={s['load_s']} replay_wall_s={s['replay_wall_s']} ms/tick={s.get('ms_per_tick')} cpu_busy={s.get('cpu_cores_busy')}")
    else:
        print(f"{name}: running demos={len(dem)} frames={fr} wall={w:.0f}s -> {fr / w if w else 0:.2f} steps/s")
