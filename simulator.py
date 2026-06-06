import numpy as np

try:
    from .controller import CascadedFlightController
    from .quadcopter import Quadcopter
except ImportError:
    from controller import CascadedFlightController
    from quadcopter import Quadcopter


class QuadcopterSimulator:
    """
    Simulates a Quadcopter trajctory tracking mission.
    """

    def __init__(self, quadcopter: Quadcopter, controller: CascadedFlightController):
        self.quad = quadcopter
        self.controller = controller

        # Waypoints queue with end time, position, and yaw target.
        self.waypoints = []
        self.current_wp_idx = 0

        # State and control history logs
        self.history = {
            "time": [],
            "state": [],  # 12-dim state vectors
            "target_pos": [],  # Selected reference position [x, y, z]
            "target_yaw": [],  # Selected reference yaw
            "motor_forces": [],  # 4-dim motor force vectors
            "control": [],  # Dictionary of control diagnostic data
        }

    def set_waypoints(self, waypoints):
        """
        Sets a list of waypoints.
        Each waypoint: (duration, [x, y, z], yaw_deg)
        """
        self.waypoints = []
        accumulated_time = 0.0
        for duration, position, yaw_deg in waypoints:
            accumulated_time += duration
            self.waypoints.append(
                {
                    "time_end": accumulated_time,
                    "pos": np.array(position, dtype=float),
                    "yaw": np.deg2rad(yaw_deg),
                }
            )
        self.current_wp_idx = 0

    def get_current_waypoint(self, current_time):
        """Gets target position and yaw based on time."""
        if not self.waypoints:
            return np.zeros(3), 0.0

        # Check if we need to advance waypoint
        while (
            self.current_wp_idx < len(self.waypoints) - 1
            and current_time > self.waypoints[self.current_wp_idx]["time_end"]
        ):
            self.current_wp_idx += 1

        wp = self.waypoints[self.current_wp_idx]
        return wp["pos"], wp["yaw"]

    def run(self, total_time=20.0, dt=0.002, control_frequency=250):
        """
        Runs the simulation loop.
        - total_time: total simulation duration (seconds)
        - dt: physics integration time step (seconds)
        - control_frequency: controller update rate (Hz)
        """
        n_steps = int(total_time / dt)
        control_steps_interval = int(1.0 / (control_frequency * dt))

        # Reset states
        self.quad.state = np.zeros(12)
        # Quadcopter starts at [0, 0, 0]
        self.controller.reset_integrators()

        # Clear logs
        for key in self.history:
            self.history[key] = []

        current_time = 0.0
        motor_forces = np.zeros(4)

        print(
            "Starting Drone simulation: "
            f"Total Time: {total_time}s, "
            f"Physics Step dt: {dt}s, "
            f"Control Rate: {control_frequency}Hz..."
        )

        for step in range(n_steps):
            # 1. Update waypoint targets
            target_pos, target_yaw = self.get_current_waypoint(current_time)

            # 2. Run controller at decimated frequency.
            if step % control_steps_interval == 0:
                motor_forces, control_log = self.controller.control(
                    self.quad.state, target_pos, target_yaw, dt * control_steps_interval
                )

            # 3. Step physics dynamics (RK4)
            self.quad.step(motor_forces, dt)

            # 4. Log telemetry history
            self.history["time"].append(current_time)
            self.history["state"].append(self.quad.state.copy())
            self.history["target_pos"].append(target_pos.copy())
            self.history["target_yaw"].append(target_yaw)
            self.history["motor_forces"].append(motor_forces.copy())
            self.history["control"].append(
                control_log.copy() if "control_log" in locals() else {}
            )

            current_time += dt

        print("Simulation complete! Processing logs...")

        # Convert history lists to numpy arrays for simpler graphing
        self.history["time"] = np.array(self.history["time"])
        self.history["state"] = np.array(self.history["state"])
        self.history["target_pos"] = np.array(self.history["target_pos"])
        self.history["target_yaw"] = np.array(self.history["target_yaw"])
        self.history["motor_forces"] = np.array(self.history["motor_forces"])

        return self.history
