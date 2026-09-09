"""The body of ``python -m experiments.ra.cli plot`` (``cli.py`` stays a dispatcher).

Output directory (D1):

* ``--out DIR``            wins over everything;
* ``--study NAME``         report tier -> ``<brain>/figures/<NAME>/``;
* exactly one ``--run-id`` -> ``<run_dir>/figures/``;
* more than one run and no ``--study`` is an error -- a figure that mixes runs
  has no single run directory to belong to.

Without ``--fail-fast`` a plot that raises :class:`MissingDataError`,
``NotImplementedError`` or :class:`RunCountError` is logged as skipped and the command still returns 0, so
``--tier debug`` stays usable on a run that did not write hourly data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from .. import paths
from . import PLOTS, MissingDataError, RunCountError, catalogue, ids, load_runs, render

logger = logging.getLogger("experiments.ra.plots")

DEFAULT_TIER = "debug"


def add_arguments(parser) -> None:
    parser.add_argument("--run-id", dest="run_ids", action="append", default=[],
                        help="a run id, a unique name prefix, or a run directory")
    parser.add_argument("--label", dest="labels", action="append", default=[],
                        help="legend label; one per --run-id, defaults to the config name")
    parser.add_argument("--plot", dest="plot_ids", action="append", default=[],
                        metavar="ID", help="plot id, repeatable (e.g. --plot O4 --plot O9)")
    parser.add_argument("--tier", default=None, choices=["debug", "report", "all"],
                        help=f"plot every plot of a tier (default: {DEFAULT_TIER})")
    parser.add_argument("--study", default=None,
                        help="write into <figures root>/<STUDY>/ instead of the run dir")
    parser.add_argument("--out", default=None, help="explicit output directory")
    parser.add_argument("--window", default=None, metavar="START:STOP")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--method", default=None)
    parser.add_argument("--block-size", dest="block_size", default=None)
    parser.add_argument("--runs-root", default=None)
    parser.add_argument("--list", action="store_true", help="print the catalogue and exit")
    parser.add_argument("--fail-fast", action="store_true",
                        help="stop at the first plot that cannot be drawn")


def parse_window(text: str | None):
    if not text:
        return None
    try:
        start, stop = str(text).split(":")
        return int(start), int(stop)
    except ValueError as exc:
        raise ValueError(f"--window expects START:STOP, got {text!r}") from exc


def resolve_out_dir(args, runs) -> Path:
    if args.out:
        return Path(args.out).expanduser()
    if args.study:
        return paths.figures_root() / str(args.study)
    if len(runs) == 1:
        return runs[0].run_dir / "figures"
    raise ValueError(
        f"{len(runs)} runs were given and no --study: a figure that compares runs has no "
        "single run directory to live in. Pass --study NAME (report figures) or --out DIR."
    )


def selected_plots(args) -> list[str]:
    if args.plot_ids:
        unknown = [p for p in args.plot_ids if p not in PLOTS]
        if unknown:
            raise ValueError(
                f"unknown plot id(s) {unknown}; known ids: {', '.join(sorted(PLOTS))}"
            )
        return list(args.plot_ids)
    return ids(tier=args.tier or DEFAULT_TIER)


def cmd_plot(args) -> int:
    if args.list:
        frame = catalogue()
        print(frame[["plot_id", "title", "tier", "phase", "needs"]].to_string(index=False))
        return 0

    if not args.run_ids:
        print("plot needs at least one --run-id", file=sys.stderr)
        return 1

    runs = load_runs(args.run_ids, runs_root=args.runs_root, labels=args.labels or None)
    try:
        out_dir = resolve_out_dir(args, runs)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    opts = {
        "window": parse_window(args.window),
        "year": args.year,
        "method": args.method,
        "block_size": args.block_size,
    }
    opts = {k: v for k, v in opts.items() if v is not None}

    written, skipped, failed = [], [], []
    for plot_id in selected_plots(args):
        try:
            png, csv = render(plot_id, runs, out_dir, **opts)
        except (MissingDataError, NotImplementedError, RunCountError) as exc:
            skipped.append((plot_id, str(exc)))
            logger.info("skipping %s: %s", plot_id, exc)
            if args.fail_fast:
                failed.append((plot_id, str(exc)))
                break
            continue
        except Exception as exc:
            failed.append((plot_id, f"{type(exc).__name__}: {exc}"))
            logger.warning("%s failed: %s", plot_id, exc, exc_info=True)
            if args.fail_fast:
                break
            continue
        written.append((plot_id, png, csv))
        logger.info("%s -> %s", plot_id, png)

    for plot_id, png, _csv in written:
        print(f"{plot_id}: {png}")
    for plot_id, reason in skipped:
        print(f"{plot_id}: skipped ({reason})")
    for plot_id, reason in failed:
        print(f"{plot_id}: FAILED ({reason})", file=sys.stderr)
    return 1 if failed else 0


def debug_figures(run_dir, cfg) -> list[Path]:
    """The ``output.figures`` hook: every debug plot the run can draw.

    Never raises: a plot with no data is skipped, exactly as ``ra plot``
    without ``--fail-fast`` does.
    """
    handles = load_runs([str(run_dir)], runs_root=None)
    out_dir = Path(run_dir) / "figures"
    written = []
    for plot_id in ids(tier="debug", phase="A"):
        try:
            png, _csv = render(plot_id, handles, out_dir)
        except (MissingDataError, NotImplementedError, RunCountError) as exc:
            logger.info("skipping %s: %s", plot_id, exc)
            continue
        except Exception as exc:
            logger.warning("%s failed: %s", plot_id, exc, exc_info=True)
            continue
        written.append(png)
    return written
