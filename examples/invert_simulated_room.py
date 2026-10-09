"""Room size from one simulated measurement, and why the source bearing must be assumed.

    python examples/invert_simulated_room.py --out examples/output/invert_simulated_room

1. Simulate a large (10.7 x 7.5 m) and a small (3.5 x 3.0 m) room with the
   image-source model, play white noise through each and add noise.
2. Identify each response with NLMS, pick the echoes and invert them; write
   the response figure, the ranked candidates and the estimate.
3. List the bearing family of the large room: every source bearing explains
   the arrivals exactly, with a slightly different room, which is why the
   bearing has to be assumed (45 degrees by default).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from room_geometry_inference import (
    Config,
    bearing_family,
    configure_logging,
    first_order_distances,
    impulse_response_table,
    plot_impulse_response,
    save_figure,
    simulate_scene,
    write_json,
)

LOGGER = logging.getLogger("invert_simulated_room")
QUICK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "quick_demo.yaml"
#: The two reference scenes: room, microphone and source (1 m off on both axes).
SCENES = {
    "large": {
        "room.size_m": [10.7, 7.5],
        "room.microphone_m": [5.35, 3.75],
        "room.source_m": [6.35, 4.75],
    },
    "small": {
        "room.size_m": [3.5, 3.0],
        "room.microphone_m": [1.75, 1.5],
        "room.source_m": [2.75, 2.5],
    },
}


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False, lineterminator="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/output/invert_simulated_room"),
        help="Output folder.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="1 s recordings (configs/quick_demo.yaml) instead of 2 s.",
    )
    args = parser.parse_args(argv)
    configure_logging("INFO")

    base = Config.from_yaml(QUICK_CONFIG) if args.quick else Config()
    base = base.override({"seed": args.seed})
    args.out.mkdir(parents=True, exist_ok=True)

    # 1-2. Simulate, identify and invert both rooms.
    rows = []
    for name, scene in SCENES.items():
        config = base.override(scene)
        result = simulate_scene(config, rng=config.seed)
        rate = config.acoustics.sample_rate
        taps = result.inversion.echo_taps
        figure = plot_impulse_response(result.weights, rate, truth=result.truth.rir, echo_taps=taps)
        save_figure(figure, args.out / f"{name}_impulse_response", config.output.figure_format)
        _write_csv(
            impulse_response_table(result.weights, rate, truth=result.truth.rir, echo_taps=taps),
            args.out / f"{name}_impulse_response.csv",
        )
        _write_csv(result.inversion.candidates, args.out / f"{name}_candidates.csv")
        estimate = result.inversion.estimate
        rows.append(
            {
                "room": name,
                "true_width_m": config.room.size_m[0],
                "true_depth_m": config.room.size_m[1],
                "width_m": estimate.width_m,
                "depth_m": estimate.depth_m,
                "source_distance_m": estimate.source_distance_m,
                "room_error_m": result.room_error_m,
                "misalignment_db": result.misalignment_db,
                "n_equivalent": estimate.n_equivalent,
            }
        )
        LOGGER.info(
            "%s room: estimated %.3f x %.3f m (error %.3f m), source %.3f m away",
            name,
            estimate.width_m,
            estimate.depth_m,
            result.room_error_m,
            estimate.source_distance_m,
        )
    _write_csv(pd.DataFrame(rows), args.out / "estimates.csv")

    # 3. The family of exact solutions over the source bearing (large room,
    #    exact first-order distances, the source 1 m off on both axes).
    distances = first_order_distances(10.7, 7.5, 5.35, 3.75, 1.0, 1.0)
    family = bearing_family(distances, bearings_deg=np.arange(5.0, 90.0, 5.0))
    _write_csv(family, args.out / "bearing_family.csv")
    write_json(
        args.out / "bearing_family_range.json",
        {
            "width_range_m": [float(family["width_m"].min()), float(family["width_m"].max())],
            "depth_range_m": [float(family["depth_m"].min()), float(family["depth_m"].max())],
        },
    )
    LOGGER.info(
        "over bearings 5-85 degrees the width spans %.3f-%.3f m and the depth %.3f-%.3f m",
        family["width_m"].min(),
        family["width_m"].max(),
        family["depth_m"].min(),
        family["depth_m"].max(),
    )
    LOGGER.info("written to %s", args.out.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
