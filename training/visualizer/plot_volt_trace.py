#!/usr/bin/env python3
"""Plot live-match telemetry written by Volt-Rust when VOLT_TRACE_DIR is set."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt


NUMERIC_FIELDS = (
    "time", "shot_score", "contact_time",
    "car_x", "car_y", "car_z", "car_vx", "car_vy", "car_vz", "boost",
    "ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy", "ball_vz",
    "car_ball_distance", "nearest_teammate_ball_distance",
    "nearest_opponent_ball_distance", "throttle", "steer", "pitch",
    "yaw", "roll",
)


def load_trace(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in NUMERIC_FIELDS:
            row[field] = float(row[field])
    return rows


def plot_player(rows: list[dict], output: Path) -> None:
    player = rows[0]["player"]
    times = [row["time"] for row in rows]
    statuses = Counter(row["status"] for row in rows)

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.plot([r["car_x"] for r in rows], [r["car_y"] for r in rows],
            label=f"Volt player {player}")
    ax.plot([r["ball_x"] for r in rows], [r["ball_y"] for r in rows],
            label="Ball", linestyle="--", alpha=0.8)
    ax.scatter(rows[0]["car_x"], rows[0]["car_y"], marker="o", label="Car start")
    ax.scatter(rows[-1]["car_x"], rows[-1]["car_y"], marker="x", label="Car end")
    ax.set_xlim(-4300, 4300)
    ax.set_ylim(-5350, 5350)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Field X")
    ax.set_ylabel("Field Y")
    ax.set_title(f"Player {player}: car and ball path")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"player_{player}_xy.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, [r["boost"] for r in rows], label="Boost")
    ax.set_xlabel("Match time (s)")
    ax.set_ylabel("Boost")
    ax.set_ylim(-2, 102)
    ax.set_title(f"Player {player}: boost")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"player_{player}_boost.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, [r["car_ball_distance"] for r in rows], label="Volt to ball")
    ax.plot(times, [r["nearest_opponent_ball_distance"] for r in rows],
            label="Nearest opponent to ball")
    ax.plot(times, [r["nearest_teammate_ball_distance"] for r in rows],
            label="Nearest teammate to ball", alpha=0.8)
    ax.set_xlabel("Match time (s)")
    ax.set_ylabel("Distance")
    ax.set_title(f"Player {player}: ball access")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"player_{player}_distances.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, [r["throttle"] for r in rows], label="Throttle")
    ax.plot(times, [r["steer"] for r in rows], label="Steer")
    ax.plot(times, [r["pitch"] for r in rows], label="Pitch")
    ax.plot(times, [r["yaw"] for r in rows], label="Yaw")
    ax.plot(times, [r["roll"] for r in rows], label="Roll")
    ax.set_xlabel("Match time (s)")
    ax.set_ylabel("Controller value")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Player {player}: controller axes")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(output / f"player_{player}_controls.png", dpi=150)
    plt.close(fig)

    with (output / f"player_{player}_summary.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"Player: {player}  Team: {rows[0]['team']}\n")
        stream.write(f"Samples: {len(rows)}  Duration: {times[-1] - times[0]:.2f}s\n")
        stream.write(f"Mean boost: {sum(r['boost'] for r in rows) / len(rows):.1f}\n")
        stream.write(f"Mean car-ball distance: {sum(r['car_ball_distance'] for r in rows) / len(rows):.1f}\n")
        stream.write(f"Jump frames: {sum(r['jump'] == 'true' for r in rows)}\n")
        stream.write(f"Boost-button frames: {sum(r['boost_button'] == 'true' for r in rows)}\n")
        stream.write("Status frames:\n")
        for status, count in statuses.most_common():
            stream.write(f"  {status or '(empty)'}: {count}\n")

    print(f"[TRACE] Player {player}: {len(rows)} samples, {times[-1] - times[0]:.1f}s")
    print(f"[TRACE] Statuses: {dict(statuses)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Directory containing volt_player_*.csv files")
    parser.add_argument("--output", type=Path,
                        help="Plot output directory (default: INPUT/plots)")
    args = parser.parse_args()

    traces = sorted(args.input.glob("volt_player_*_team_*.csv"))
    if not traces:
        raise SystemExit(
            f"No Volt trace CSV files found in {args.input}. "
            "Set VOLT_TRACE_DIR before launching the RLBot match."
        )

    output = args.output or args.input / "plots"
    output.mkdir(parents=True, exist_ok=True)

    for path in traces:
        rows = load_trace(path)
        if rows:
            plot_player(rows, output)

    print(f"[TRACE] Plots and summaries written to: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
