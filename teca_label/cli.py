"""`teca-label` on the command line.

    teca-label run [--codebook NAME] [--project DIR]   one tick of the standing loop
    teca-label apply --codebook NAME [--project DIR]   accept a reviewed pending revision
    teca-label status [--project DIR]                  each codebook's version, revisions, pending, last run

`run` exits 10 when a proposal awaits review — the branch a scheduler forks on.
"""
import argparse
import json
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="teca-label", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="one tick per codebook: label new rows, check drift, "
                                     "write a proposal when it's due (exit 10 if one awaits)")
    run.add_argument("--codebook", help="one codebook (default: every codebook in teca-label.toml)")
    run.add_argument("--project", default=".", help="directory containing teca-label.toml (default: .)")
    apply_cmd = sub.add_parser("apply", help="accept a codebook's reviewed pending revision")
    apply_cmd.add_argument("--codebook", required=True, help="the codebook whose pending file to apply")
    apply_cmd.add_argument("--project", default=".", help="directory containing teca-label.toml (default: .)")
    status = sub.add_parser("status", help="every codebook at a glance")
    status.add_argument("--project", default=".", help="directory containing teca-label.toml (default: .)")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "run":
            sys.exit(_run(args.codebook, args.project))
        elif args.cmd == "apply":
            _apply(args.codebook, args.project)
        elif args.cmd == "status":
            _status(args.project)
    except (FileNotFoundError, ValueError, ImportError, ConnectionError) as exc:
        # config, missing-extra, and bad-input errors are the user's to fix: one line, no traceback
        raise SystemExit(f"teca-label: {exc}") from None


def _codebooks(project: str, only: str | None = None) -> dict:
    from teca_label import runner
    codebooks = runner.load_project(project)
    if only is None:
        return codebooks
    if only not in codebooks:
        raise SystemExit(f"teca-label: no codebook '{only}' in teca-label.toml — have: "
                         f"{', '.join(sorted(codebooks))}")
    return {only: codebooks[only]}


def _run(only: str | None, project: str) -> int:
    from teca_label import runner
    any_proposed = False
    for name, spec in _codebooks(project, only).items():
        summary = runner.tick(name, spec)
        print(json.dumps(summary, indent=2, default=str))
        any_proposed = any_proposed or summary["proposed"]
    return runner.TICK_PROPOSED if any_proposed else 0


def _apply(only: str, project: str) -> None:
    from teca_label import runner
    from teca_label.core import Codebook
    spec = _codebooks(project, only)[only]
    cb = Codebook.load(spec["path"])
    cb.apply()                                    # bare apply: consume the reviewed pending file
    runner.pending_md_path(cb).unlink(missing_ok=True)
    print(f"{only} → v{cb.version}")
    print(cb.history()[-1])


def _status(project: str) -> None:
    import os
    from teca_label.core import Codebook
    for name, spec in _codebooks(project).items():
        cb = Codebook.load(spec["path"])
        report = cb.staleness()
        pending = " · PENDING REVIEW" if cb.pending_path.exists() else ""
        print(f"{name:24} v{cb.version} · {report['active_categories']} categories · "
              f"{report['revisions']} revisions · last {report['last_revision_at'] or 'never'}"
              f"{pending}")
        print(f"{'':24} {_run_line(spec, os.environ.get(spec['dsn_env'], ''))}")


def _run_line(spec: dict, dsn: str) -> str:
    """One line on the last ledger row — or why there isn't one. Never raises:
    status must work offline, before any tick, and without the extra."""
    if not dsn:
        return f"last run: unknown ({spec['dsn_env']} not set)"
    try:
        from teca_label import sources
        run = sources.last_run(dsn, spec["name"], spec["runs_table"])
    except Exception as unreachable:
        return f"last run: unreachable ({str(unreachable).strip().splitlines()[0][:60]})"
    if run is None:
        return "last run: none recorded"
    line = (f"last run: {run['status']} · {str(run['run_at'])[:16]} · "
            f"labeled {run['rows_labeled'] if run['rows_labeled'] is not None else '?'}")
    if run["gaps"]:
        line += f" · gaps {run['gaps']}"
    if run["other_share"] is not None and run["rows_labeled"]:
        line += f" · other {100 * run['other_share']:.1f}%"
    if run["status"] == "running":
        line += "   <- in flight, or a crashed tick"
    return line


if __name__ == "__main__":
    main()
