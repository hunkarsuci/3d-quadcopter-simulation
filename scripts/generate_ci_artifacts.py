import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from animate import animate_copter_3d  # noqa: E402
from controller import CascadedFlightController  # noqa: E402
from quadcopter import Quadcopter  # noqa: E402
from simulator import QuadcopterSimulator  # noqa: E402


def run_demo():
    quad = Quadcopter()
    controller = CascadedFlightController(quad.mass, quad.g, quad.L, quad.c_tf)
    sim = QuadcopterSimulator(quad, controller)
    sim.set_waypoints(
        [
            (0.4, [0.0, 0.0, 0.5], 0.0),
            (0.4, [0.4, 0.0, 0.5], 30.0),
            (0.4, [0.4, 0.4, 0.7], 60.0),
        ]
    )
    return sim.run(total_time=1.2, dt=0.004, control_frequency=125)


def save_telemetry_plot(history, output_path):
    time = history["time"]
    state = history["state"]
    target_pos = history["target_pos"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time, state[:, 2], label="Z actual")
    axes[0].plot(time, target_pos[:, 2], "--", label="Z target")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, state[:, 0], label="X actual")
    axes[1].plot(time, state[:, 1], label="Y actual")
    axes[1].plot(time, target_pos[:, 0], "--", label="X target")
    axes[1].plot(time, target_pos[:, 1], "--", label="Y target")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position (m)")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="ci_artifacts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    history = run_demo()

    save_telemetry_plot(history, os.path.join(args.output_dir, "telemetry.png"))
    animate_copter_3d(
        history,
        save_path=os.path.join(args.output_dir, "flight.gif"),
    )
    plt.close("all")


if __name__ == "__main__":
    main()
