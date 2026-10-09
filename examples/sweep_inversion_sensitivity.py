"""How timing offsets, a wrong source bearing, source displacement and noise affect the estimate.

    python examples/sweep_inversion_sensitivity.py --out examples/output/sweep_inversion_sensitivity

1. Around the large reference room (10.7 x 7.5 m), add constant offsets to
   every arrival, move the source around the microphone and along the axes,
   and invert the arrivals under the assumed 45-degree bearing.
2. Run the complete chain (simulated recording, NLMS, echo picking,
   inversion) at several signal-to-noise ratios, several noise realizations
   each.
3. Write the tidy table, its summary and one figure (with its table) per sweep.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from room_geometry_inference import (
    Config,
    configure_logging,
    plot_sensitivity,
    run_sweeps,
    save_figure,
    sensitivity_table,
    summarize_sweeps,
)

LOGGER = logging.getLogger("sweep_inversion_sensitivity")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/sweep_inversion_sensitivity"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer sweep points, 2 noise trials and 1 s recordings (configs/quick_demo.yaml).",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    args.out.mkdir(parents=True, exist_ok=True)

    # 1-2. Every configured sweep, in one tidy table.
    table = run_sweeps(config, rng=config.seed)
    summary = summarize_sweeps(table)
    _write_csv(table, args.out / "sensitivity.csv")
    _write_csv(summary, args.out / "sensitivity_summary.csv")

    # 3. One figure and its table per sweep.
    for sweep in config.sensitivity.sweeps:
        stem = args.out / f"sensitivity_{sweep}"
        save_figure(plot_sensitivity(summary, sweep), stem, config.output.figure_format)
        _write_csv(sensitivity_table(summary, sweep), stem.with_name(f"{stem.name}.csv"))
    for row in summary.itertuples(index=False):
        LOGGER.info(
            "%s %s=%g: size error %.3f m, success %.0f%%",
            row.sweep,
            row.parameter,
            row.value,
            row.room_error_mean_m,
            100.0 * row.success_rate,
        )
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
