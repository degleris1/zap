"""
Sherlock SLURM launcher for the DC-placement study family across the rebuilt
ERCOT / WECC-1493 / Eastern networks (and the WECC-490 reference).

Why a new launcher: experiments/{solve,plan,conic_solve}/launch.py are
Perlmutter-flavored (--qos=shared, --constraint=cpu|gpu&hbm80g) and wired to
experiments/*/runner.py, not to the standalone development/dc_*.py CLIs. This
emits Sherlock sbatch scripts for the dc_*.py family with the agreed conventions:

  * env: source /scratch/users/gfw/zap-venv/bin/activate (uv-built, CPU torch,
    zap installed editable so repo edits are live)
  * read networks from $SCRATCH; write outputs to $GROUP_HOME/gfw/dc_sweep
  * REPLAY the 490 canonical invocations -- only deltas are --network,
    penetration-scaled budgets/fleets, the frozen per-net manifest, uniform
    siting (--land-cost ''), and --outdir
  * a per-net `prep` job freezes the cleaning manifest; studies that consume it
    run afterok on prep
  * partition `normal` (Sherlock cap is 7d, so no --qos=long needed)

Penetration (from pilot_network.py on the FIXED builds, B = 10 * load/79):
  texas   32 GW -> B 4.1   (f 0.41)
  western 70 GW -> B 8.9   (f 0.89)
  eastern ~321  -> B ~40   (f ~4.0)   [confirm from the pilot log before launch]

Usage:
  python development/launch_dc_sweep.py --dry-run                 # write scripts, don't submit
  python development/launch_dc_sweep.py --nets texas --studies study,wall
  python development/launch_dc_sweep.py                           # write + sbatch everything
"""
import argparse
import os
import subprocess

VENV = "/scratch/users/gfw/zap-venv/bin/activate"
REPO = "/home/users/gfw/zap"
RES = "/scratch/users/gfw/pypsa-usa/resources/Default"
SWEEP = "/home/groups/ramr/gfw/dc_sweep"
MANIFEST_DIR = "development/results/cleaning"        # repo-relative (committed with results)
CACHE_DIR = "development/results/placement_study"

# Per-network config. budgets = study ladder ([3,4,5,6]*f); fleets = wall/frontier
# ladder ([6,10]*f). mem sized to the .nc; cand_stride thins deliverable candidates
# on the big nets to keep the LP tractable.
# time_deliverable is SEPARATE from time_big: the deliverable-frontier headroom loop is the
# long pole (the original eastern deliverable logged 38.4h > its 24h time_big and truncated at
# 3/5 kappa). dc_deliverable_frontier.py is now resumable (h_firm cache + kappa-skip), so a
# 2-day eastern walltime should finish in one shot; if not, resubmit the same job to continue.
NETS = {
    "texas":   dict(nc=f"{RES}/texas/elec_s500_c500.nc",     mem="16G", cpus=4,
                    budgets="1.5,2,2.5,3",   fleets="2.5,4",   cand_stride=1, spread_stride=1,
                    time_big="08:00:00", time_deliverable="08:00:00"),
    "western": dict(nc=f"{RES}/western/elec_s1493_c1493.nc", mem="32G", cpus=4,
                    budgets="3,4,5,6",       fleets="6,10",    cand_stride=2, spread_stride=1,
                    time_big="12:00:00", time_deliverable="12:00:00"),
    "eastern": dict(nc=f"{RES}/eastern/elec_s3000_c3000.nc", mem="64G", cpus=8,
                    budgets="12,16,20,24",   fleets="24,41",   cand_stride=4, spread_stride=2,
                    time_big="24:00:00", time_deliverable="2-00:00:00"),
}

# Networks that get the flat/training robustness arm (separate single-workload jobs). texas
# already has both workloads complete; western + eastern still need training (Decision #8).
TRAIN_NETS = {"western", "eastern"}

# Decision #8: inference is the primary workload on ALL nets; the flat/training profile
# is a robustness arm on EASTERN ONLY. Both frontier scripts now save per-workload
# incrementally, so a single-workload job can't lose results to a timeout/preemption.

# Studies that need the frozen manifest run afterok on prep. `wall` re-detects bad
# buses per threshold by design (its robustness axis), so it does NOT take a manifest
# and does NOT depend on prep.
def study_cmds(net, cfg):
    man = f"{MANIFEST_DIR}/bad_buses_{net}_load1p0.json"
    cache = f"{CACHE_DIR}/usable_nodes_{net}.json"
    od = lambda s: f"{SWEEP}/results/{s}/{net}"
    cmds = {
        # name        needs_prep  walltime      command
        "study": (True, cfg["time_big"],
                  f"python development/dc_placement_study.py --network {cfg['nc']} "
                  f"--load-scale 1.0 --budgets {cfg['budgets']} --n-fleets 24 "
                  f"--bad-buses-json {man} --usable-json {cache} --land-cost '' "
                  f"--outdir {od('study')}"),
        "wall": (False, cfg["time_big"],
                 f"python development/dc_wall_robustness.py --network {cfg['nc']} "
                 f"--fleets {cfg['fleets']} --n-pools 24 --pool-size 40 --land-cost '' "
                 f"--tag {net} --outdir {od('wall')}"),
        "spread": (True, cfg["time_big"],
                   f"python development/dc_spread_frontier.py --network {cfg['nc']} "
                   f"--n-snaps 12 --n-fleets 12 --node-stride {cfg['spread_stride']} "
                   f"--bad-buses-json {man} --land-cost '' --workloads inference "
                   f"--tag {net} --outdir {od('spread')}"),
        "deliverable": (True, cfg["time_deliverable"],
                        f"python development/dc_deliverable_frontier.py --network {cfg['nc']} "
                        f"--n-snaps 12 --kappa 0.5 1 2 4 8 --skip-n1 --cand-stride {cfg['cand_stride']} "
                        f"--bad-buses-json {man} --land-cost '' --workloads inference "
                        f"--tag {net} --outdir {od('deliverable')}"),
    }
    # Training robustness arm (separate single-workload jobs) for the nets that still need it.
    # deliverable is resumable: a timed-out training run resumes on resubmit from its h_firm
    # cache + completed kappa, same as inference.
    if net in TRAIN_NETS:
        cmds["spread_train"] = (True, cfg["time_big"],
                   f"python development/dc_spread_frontier.py --network {cfg['nc']} "
                   f"--n-snaps 12 --n-fleets 12 --node-stride {cfg['spread_stride']} "
                   f"--bad-buses-json {man} --land-cost '' --workloads training "
                   f"--tag {net}_train --outdir {od('spread')}")
        cmds["deliverable_train"] = (True, cfg["time_deliverable"],
                        f"python development/dc_deliverable_frontier.py --network {cfg['nc']} "
                        f"--n-snaps 12 --kappa 0.5 1 2 4 8 --skip-n1 --cand-stride {cfg['cand_stride']} "
                        f"--bad-buses-json {man} --land-cost '' --workloads training "
                        f"--tag {net}_train --outdir {od('deliverable')}")
    return cmds


SBATCH_TMPL = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --output={SWEEP}/slurm/{job}_%j.out
#SBATCH --partition={partition}
{qos}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
{dep}
set -euo pipefail
source {VENV}
cd {REPO}
export OMP_NUM_THREADS={cpus}
echo "host $(hostname) | job {job} | $(date)"
{cmd}
echo "=== DONE {job} ==="
"""


def write_script(job, partition, cpus, mem, time, cmd, dep_jobid=None, qos=None):
    dep = f"#SBATCH --dependency=afterok:{dep_jobid}" if dep_jobid else ""
    qos_line = f"#SBATCH --qos={qos}" if qos else ""
    body = SBATCH_TMPL.format(job=job, SWEEP=SWEEP, partition=partition, qos=qos_line,
                              cpus=cpus, mem=mem, time=time, dep=dep, VENV=VENV, REPO=REPO, cmd=cmd)
    path = f"{SWEEP}/scripts/{job}.sbatch"
    with open(path, "w") as f:
        f.write(body)
    return path


def submit(path, dry):
    if dry:
        print(f"  [dry-run] would sbatch {path}")
        return None
    out = subprocess.check_output(["sbatch", path], text=True).strip()
    jid = out.split()[-1]
    print(f"  submitted {os.path.basename(path)} -> job {jid}")
    return jid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default="texas,western,eastern")
    ap.add_argument("--studies", default="study,wall,spread,deliverable")
    ap.add_argument("--partition", default="normal")
    ap.add_argument("--qos", default=None,
                    help="QoS, e.g. high_p to use the group's owner allocation on -p owners "
                         "(routes around the normal-QoS per-account 512-CPU cap).")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(f"{SWEEP}/scripts", exist_ok=True)
    os.makedirs(f"{SWEEP}/slurm", exist_ok=True)
    nets = [n for n in args.nets.split(",") if n]
    studies = [s for s in args.studies.split(",") if s]

    for net in nets:
        cfg = NETS[net]
        print(f"\n### {net}  ({os.path.basename(cfg['nc'])})  budgets={cfg['budgets']} fleets={cfg['fleets']}")
        # prep: freeze manifest (needed by study/spread/deliverable)
        man = f"{MANIFEST_DIR}/bad_buses_{net}_load1p0.json"
        prep_cmd = f"python development/dc_freeze_manifest.py --network {cfg['nc']} --out {man}"
        prep_path = write_script(f"prep_{net}", args.partition, cfg["cpus"], cfg["mem"],
                                 "00:30:00", prep_cmd, qos=args.qos)
        need_prep = any(study_cmds(net, cfg)[s][0] for s in studies if s in study_cmds(net, cfg))
        prep_jid = submit(prep_path, args.dry_run) if need_prep else None

        for s in studies:
            cmds = study_cmds(net, cfg)
            if s not in cmds:
                print(f"  (skip unknown study '{s}')")
                continue
            needs_prep, walltime, cmd = cmds[s]
            dep = prep_jid if needs_prep else None
            path = write_script(f"{s}_{net}", args.partition, cfg["cpus"], cfg["mem"],
                                walltime, cmd, dep_jobid=dep, qos=args.qos)
            submit(path, args.dry_run)

    print("\nDone. Monitor with: squeue --me ; tail -f", f"{SWEEP}/slurm/*.out")


if __name__ == "__main__":
    main()
