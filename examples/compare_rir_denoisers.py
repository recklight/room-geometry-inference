"""Compare denoisers on identified impulse responses of rooms they have not seen.

    python examples/compare_rir_denoisers.py --out examples/output/compare_rir_denoisers

1. Write synthetic measurements at a low signal-to-noise ratio (5 dB), so
   the identified responses carry a clear noise floor.
2. Pair every identified response with the true one, simulated from the
   geometry stored in dataset.csv.
3. Cross-validate the denoisers with folds grouped by room (each room is
   denoised by a model that never saw it) and write the per-response MSE,
   its summary and a box plot. With the dl extra installed, the PyTorch
   deep autoencoder joins the comparison (not with --quick).

All responses here are simulated; the MSE ranks the denoisers on synthetic
rooms only.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

import pandas as pd

from room_geometry_inference import (
    Config,
    compare_denoisers,
    configure_logging,
    denoising_pairs,
    denoising_scores_table,
    denoising_summary,
    plot_denoising_scores,
    save_figure,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_rir_denoisers")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_rir_denoisers"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small data and networks (configs/quick_demo.yaml), no deep autoencoder.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    methods = ["identity", "linear", "pca", "mlp"]
    if importlib.util.find_spec("torch") is not None and not args.quick:
        methods.append("deep_autoencoder")
    config = config.override(
        {"seed": args.seed, "synthetic.snr_db": 5.0, "denoising.methods": methods}
    )
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Noisy synthetic measurements.
    index = write_synthetic_dataset(
        args.out / "data", config.synthetic, rng=config.seed, acoustics=config.acoustics
    )

    # 2. Identified and true responses.
    noisy, clean, groups = denoising_pairs(index, config)

    # 3. Room-grouped cross-validation of every denoiser.
    settings = config.denoising
    scores = compare_denoisers(
        noisy,
        clean,
        groups,
        methods=settings.methods,
        architectures=settings.architectures,
        n_splits=config.evaluation.n_splits,
        random_state=config.seed,
        options=settings.options(),
    )
    summary = denoising_summary(scores)
    _write_csv(denoising_scores_table(scores), args.out / "denoising_scores.csv")
    _write_csv(summary, args.out / "denoising_summary.csv")
    save_figure(plot_denoising_scores(scores), args.out / "denoising_scores", "png")
    for row in summary.itertuples(index=False):
        LOGGER.info(
            "%s: %.2f +/- %.2f dB (gain %.2f dB over the noisy response)",
            row.method,
            row.mean_db,
            row.std_db,
            row.gain_db,
        )
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
