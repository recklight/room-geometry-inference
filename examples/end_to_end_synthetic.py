"""End to end on synthetic data: synthesize, prepare, cross-validate, train, predict.

    python examples/end_to_end_synthetic.py --out examples/output/end_to_end_synthetic

1. Write synthetic measurements of large and small rooms (two-channel WAV
   files and dataset.csv) with the same on-disk layout as real data.
2. Identify the impulse response of every measurement with NLMS and describe
   it by its energy decay curve.
3. Cross-validate the room classifier with folds grouped by room, so every
   test room is unseen.
4. Train on every measurement and save the model.
5. Load the model and classify one measurement of each class.

On simulated rooms the UAR only tells me that the chain runs end to end, not
how well recorded rooms would be classified.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from room_geometry_inference import (
    Config,
    configure_logging,
    cross_validate,
    items_table,
    load_model,
    predict_inputs,
    prepare_features,
    save_features,
    save_model,
    train_model,
    write_json,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("end_to_end_synthetic")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/end_to_end_synthetic"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="8 rooms, 16 recordings of 1 s and 4 folds (configs/quick_demo.yaml).",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    config = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    config = config.override({"seed": args.seed})
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Synthetic measurements and their dataset.csv index.
    index = write_synthetic_dataset(
        args.out / "data", config.synthetic, rng=config.seed, acoustics=config.acoustics
    )

    # 2. Identified responses described by their decay curves.
    features = prepare_features(index, config)
    save_features(args.out / "features.npz", features)
    _write_csv(items_table(features), args.out / "items.csv")
    LOGGER.info("%d measurements, %d features each", features.n_items, features.X.shape[1])

    # 3. Cross-validation grouped by room.
    result = cross_validate(features, config)
    write_json(args.out / "metrics.json", result.metrics)
    _write_csv(result.folds, args.out / "folds.csv")
    _write_csv(result.predictions, args.out / "predictions.csv")
    summary = result.metrics["summary"]["uar"]
    LOGGER.info("UAR on unseen rooms: %.3f +/- %.3f", summary["mean"], summary["std"])

    # 4. One model on every measurement.
    model_path = save_model(
        args.out / "model.joblib", train_model(features, config), config, features.classes
    )

    # 5. Reload it and classify the first measurement of every class.
    bundle = load_model(model_path)
    data = pd.read_csv(index)
    chosen = data.groupby("label", sort=True)["path"].first()
    table = predict_inputs([index.parent / relative for relative in chosen], bundle)
    _write_csv(table, args.out / "new_predictions.csv")
    for label, predicted in zip(chosen.index, table["predicted"], strict=True):
        LOGGER.info("a %s-room measurement was classified as %s", label, predicted)
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
