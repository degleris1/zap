# DC multi-network sweep — jobs to complete the full result set

Status as of 2026-06-17. The recast trio (**wall** + **spread** + **deliverable**), each ×
{inference, training}, on the three rebuilt nets (texas-500 / western-1493 / eastern-3000) is
the citable set. The first SLURM sweep reproduced the distribute-beats-concentrate headline on
all three, but four jobs are missing or truncated. Everything here runs from the repo root on
Sherlock via `development/launch_dc_sweep.py` (writes sbatch to `$GROUP_HOME/gfw/dc_sweep`,
reads nets from `$SCRATCH`, env `source /scratch/users/gfw/zap-venv/bin/activate`).

## Coverage (net × study × workload)

```
            wall   spread (inf/train)   deliverable (inf/train)
 texas       OK     OK   / OK            OK(5/5) / OK(5/5)          COMPLETE
 western      OK     OK   / MISSING       OK(5/5) / MISSING
 eastern      OK     OK   / OK            3of5    / 3of5            (truncated at kappa=2)
```

`dc_placement_study` ("study") arm: **deliberately dropped** — it's the older directional
Q1/Q2 (un-gated metric), superseded by the gated spread/deliverable/wall per the locked paper
decisions. The launcher still emits it as an option; just don't pass `--studies study`.

## Jobs to run

Each job auto-creates its `prep_<net>` manifest-freeze dependency. Submit with the launcher;
drop `--dry-run` to actually sbatch. `dc_deliverable_frontier.py` is now **resumable** (h_firm
cache + completed-kappa skip), so a timed-out deliverable job continues forward on resubmit.

**[1] eastern deliverable — inference, complete κ=4,8 (resume)**  ·  walltime 2-00:00:00
```
python development/launch_dc_sweep.py --nets eastern --studies deliverable
```

**[2] eastern deliverable — training, complete κ=4,8 (resume)**  ·  walltime 2-00:00:00
```
python development/launch_dc_sweep.py --nets eastern --studies deliverable_train
```

**[3] western spread — training (never run)**  ·  walltime 12:00:00
```
python development/launch_dc_sweep.py --nets western --studies spread_train
```

**[4] western deliverable — training (never run)**  ·  walltime 12:00:00
```
python development/launch_dc_sweep.py --nets western --studies deliverable_train
```

Jobs [1]+[2] and [3]+[4] can be batched per net:
`--nets eastern --studies deliverable,deliverable_train` and
`--nets western --studies spread_train,deliverable_train`.

### Resume-seeding (do this BEFORE submitting [1] and [2])

So the eastern resume **skips the ~38 h headroom loop** and only computes κ=4,8, copy the
pulled-back (partial) result JSONs into Sherlock's deliverable outdir first — the resumable
script seeds `h_firm` from the embedded values and reuses the κ=0.5,1,2 rows:
```
DST=/home/groups/ramr/gfw/dc_sweep/results/deliverable/eastern
scp development/results/deliverable_frontier/deliverable_frontier_eastern.json       gfw@login.sherlock.stanford.edu:$DST/
scp development/results/deliverable_frontier/deliverable_frontier_eastern_train.json gfw@login.sherlock.stanford.edu:$DST/
```
On start the job logs `h_firm seeded from result JSON (N candidates) -> skipping headroom loop`
and `resumed 3 completed kappa; computed 2 new this run`. If you skip the copy it still works —
it just recomputes h_firm from scratch (checkpointed every 25 candidates). If a job times out,
resubmit the identical sbatch; it continues forward (SLURM `--requeue` does NOT fire on a
walltime TIMEOUT, so resubmit manually).

### Pull results back

After the jobs finish, rsync the small artifacts back into the repo and commit (same as the
first sweep):
```
rsync -avz --include='*/' --include='*.json' --include='*.png' --exclude='*' \
  gfw@login.sherlock.stanford.edu:/home/groups/ramr/gfw/dc_sweep/results/ \
  development/results/   # then mv/flatten into spread_frontier/ deliverable_frontier/ as needed
```
Re-run the consistency read and confirm: eastern deliverable is 5/5 κ with `sanity_checks`
populated (was `null`), western has training, and the κ=4,8 rows extend the monotone LP-GW
trend (texas plateaus ~39 GW at κ≥4; eastern was still rising at κ=2).

## Quality follow-ups (not blocking the full set)

**[Q1] Eastern cleanliness + @41 wall tie.** Eastern's base isn't fully clean on the 10-bus
default manifest (residual base shed median 0.24 % / max 1.97 %), the prime suspect for the
eastern @41 GW `conc_feas == dist_feas` wall tie (the one place the headline does not separate).
- The strict 15-bus list is already computed:
  `wall_robustness_eastern.json → threshold_axis.strict.bad_buses`. Either lift it directly, or
  extend `dc_freeze_manifest.py` to expose the `find_bad_buses` thresholds
  (`hi_cap`/`lo_cap`/`shed_frac`, see `dc_placement_study.py:182`) and freeze a strict manifest.
- Re-pilot eastern on the strict manifest (`development/diagnostics/pilot_network.py`); check
  whether base shed → ~0 and whether the @41 tie dissolves.
- Re-run eastern wall + spread + deliverable on the strict manifest **only if** it materially
  moves results (this discards the current eastern wall/spread, so gate it on the pilot).

## Notes
- Penetration-matched ladders (already in the launcher): texas B 1.5–3 / fleets 2.5,4;
  western B 3–6 / fleets 6,10; eastern B 12–24 / fleets 24,41.
- All new builds carry **2022 weather relabeled to 2025 horizons** (the 490 reference is 2023) —
  a writeup caveat for cross-net *level* comparisons; within-net claims are unaffected.
