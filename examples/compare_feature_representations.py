"""Compare four descriptions of the identified response for classifying unseen rooms.

    python examples/compare_feature_representations.py

1. Write synthetic measurements of large and small rooms: several rooms per
   class, several placements per room.
2. Identify every response once, then describe it four ways: the raw
   magnitude, the energy decay curve, the echo arrival times and the
   intrinsic-mode energies.
3. Cross-validate every classifier on every representation it can read (the
   template matcher reads only the magnitude) with folds grouped by room, and
   write the UAR table and a bar chart. The rooms are simulated; on recorded
   rooms the ranking may not hold.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from room_geometry_inference import (
    MODEL_NAMES,
    REPRESENTATIONS,
    Config,
    FeatureSet,
    configure_logging,
    cross_validate,
    load_dataset_index,
    measurement_response,
    new_figure,
    response_features,
    save_figure,
    write_synthetic_dataset,
)

LOGGER = logging.getLogger("compare_feature_representations")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/compare_feature_representations"),
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

    # 1. Synthetic measurements.
    index_path = write_synthetic_dataset(
        args.out / "data", config.synthetic, rng=config.seed, acoustics=config.acoustics
    )
    index = load_dataset_index(index_path)

    # 2. One identification per measurement, four descriptions of it.
    responses = [measurement_response(path, config) for path in index["path"]]
    rows = []
    for representation in REPRESENTATIONS:
        described = config.override({"features.representation": representation})
        features = FeatureSet(
            X=np.vstack([response_features(response, described) for response in responses]),
            y=index["label"].to_numpy(dtype=str),
            groups=index["group"].to_numpy(dtype=str),
            paths=index["source"].to_numpy(dtype=str),
            config=described.to_dict(),
        )
        # 3. Every classifier that can read this representation.
        for model in MODEL_NAMES:
            if model == "template" and representation != "magnitude":
                continue
            result = cross_validate(features, described.override({"model.name": model}))
            uar = result.metrics["summary"]["uar"]
            rows.append(
                {
                    "representation": representation,
                    "model": model,
                    "uar_mean": uar["mean"],
                    "uar_std": uar["std"],
                    "pooled_uar": result.metrics["pooled"]["uar"],
                }
            )
            LOGGER.info(
                "%s + %s: UAR %.3f +/- %.3f", representation, model, uar["mean"], uar["std"]
            )
    table = pd.DataFrame(rows)
    _write_csv(table, args.out / "representation_comparison.csv")

    figure = new_figure(7.0, 3.6)
    axes = figure.add_subplot()
    labels = [f"{row.representation}\n{row.model}" for row in table.itertuples(index=False)]
    axes.bar(range(len(table)), table["uar_mean"], yerr=table["uar_std"].fillna(0.0), capsize=3)
    axes.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    axes.set_xticks(range(len(table)), labels, fontsize=7)
    axes.set_ylabel("UAR on unseen rooms")
    axes.set_ylim(0.0, 1.05)
    axes.set_title("Room-size classification by response representation")
    save_figure(figure, args.out / "representation_comparison", "png")
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
