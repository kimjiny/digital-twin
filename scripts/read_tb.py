import glob, os
from tensorboard.backend.event_processing import event_accumulator

base = os.path.expanduser("~/jy/isaaclab/logs/skrl/human_nav_direct")
runs = sorted(glob.glob(os.path.join(base, "*")))
run = runs[-1]
ev = glob.glob(os.path.join(run, "events.out.tfevents.*"))[0]
ea = event_accumulator.EventAccumulator(ev, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
print("RUN:", os.path.basename(run))
print("TAGS:", tags)
# find reward-related tag
for t in tags:
    if "reward" in t.lower() and "mean" in t.lower():
        vals = ea.Scalars(t)
        xs = [v.step for v in vals]
        ys = [v.value for v in vals]
        n = len(ys)
        if n:
            first = sum(ys[:max(1, n//10)]) / max(1, n//10)
            last = sum(ys[-max(1, n//10):]) / max(1, n//10)
            print(f"TAG[{t}] n={n} first_avg={first:.3f} last_avg={last:.3f} min={min(ys):.3f} max={max(ys):.3f}")
