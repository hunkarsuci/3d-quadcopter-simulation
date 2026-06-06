import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def plot_flight_telemetry(history, show=True):
    """Plots flight telemetry data (Position, Euler angles, motor outputs)."""
    t = history["time"]
    state = history["state"]
    target_pos = history["target_pos"]
    target_yaw = history["target_yaw"]
    motor_forces = history["motor_forces"]

    # Extract state variables
    x, y, z = state[:, 0], state[:, 1], state[:, 2]
    vx, vy, vz = state[:, 3], state[:, 4], state[:, 5]
    phi, theta, psi = (
        np.rad2deg(state[:, 6]),
        np.rad2deg(state[:, 7]),
        np.rad2deg(state[:, 8]),
    )
    p, q, r = (
        np.rad2deg(state[:, 9]),
        np.rad2deg(state[:, 10]),
        np.rad2deg(state[:, 11]),
    )

    # Extract targets
    x_trg, y_trg, z_trg = target_pos[:, 0], target_pos[:, 1], target_pos[:, 2]
    yaw_trg = np.rad2deg(target_yaw)

    fig = plt.figure(figsize=(14, 10))

    # 1. 3D Position tracking
    plt.subplot(3, 2, 1)
    plt.plot(t, x, "r-", label="X Actual")
    plt.plot(t, x_trg, "r--", alpha=0.7, label="X Target")
    plt.plot(t, y, "g-", label="Y Actual")
    plt.plot(t, y_trg, "g--", alpha=0.7, label="Y Target")
    plt.plot(t, z, "b-", label="Z Actual")
    plt.plot(t, z_trg, "b--", alpha=0.7, label="Z Target")
    plt.grid(True)
    plt.title("Position Tracking (m)")
    plt.xlabel("Time (s)")
    plt.legend()

    # 2. Attitude Euler Angles
    plt.subplot(3, 2, 2)
    plt.plot(t, phi, "r-", label="Roll (phi)")
    plt.plot(t, theta, "g-", label="Pitch (theta)")
    plt.plot(t, psi, "b-", label="Yaw (psi) Actual")
    plt.plot(t, yaw_trg, "b--", alpha=0.7, label="Yaw Target")
    plt.grid(True)
    plt.title("Euler Angles Attitude (deg)")
    plt.xlabel("Time (s)")
    plt.legend()

    # 3. Velocities
    plt.subplot(3, 2, 3)
    plt.plot(t, vx, "r-", label="Vx")
    plt.plot(t, vy, "g-", label="Vy")
    plt.plot(t, vz, "b-", label="Vz")
    plt.grid(True)
    plt.title("Linear Velocities (m/s)")
    plt.xlabel("Time (s)")
    plt.legend()

    # 4. Body Rates
    plt.subplot(3, 2, 4)
    plt.plot(t, p, "r-", label="p (roll rate)")
    plt.plot(t, q, "g-", label="q (pitch rate)")
    plt.plot(t, r, "b-", label="r (yaw rate)")
    plt.grid(True)
    plt.title("Body Rates (deg/s)")
    plt.xlabel("Time (s)")
    plt.legend()

    # 5. Motor Outputs (Forces)
    plt.subplot(3, 2, 5)
    for i in range(4):
        plt.plot(t, motor_forces[:, i], label=f"Motor {i+1}")
    plt.grid(True)
    plt.title("Motor Thrust Forces (N)")
    plt.xlabel("Time (s)")
    plt.legend()

    plt.tight_layout()
    if show:
        plt.show()

    return fig


def animate_copter_3d(history, arm_length=0.25, save_path=None):
    """
    Creates an interactive 3D animation of the quadcopter flying.
    Plots the structural arm axes (X-configuration) and trailing flights.
    """
    # Downsample telemetry frame rates to keep animation running smoothly at ~30 FPS
    # Physics is run at 500 Hz (dt=0.002). 500 / 15 steps is about 33 FPS.
    step_size = 15
    t = history["time"][::step_size]
    states = history["state"][::step_size]
    targets = history["target_pos"][::step_size]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Quadcopter structures
    # Arm length adjusted matrix transforms
    # Body Frame axes in X-Configuration:
    # Motor 1: (+L_x, +L_y), Motor 2: (+L_x, -L_y)
    # Motor 3: (-L_x, -L_y), Motor 4: (-L_x, +L_y)
    l_arm = arm_length / np.sqrt(2)
    arm13_body = np.array(
        [[l_arm, l_arm, 0], [-l_arm, -l_arm, 0]]  # Motor 1  # Motor 3
    ).T

    arm24_body = np.array(
        [[l_arm, -l_arm, 0], [-l_arm, l_arm, 0]]  # Motor 2  # Motor 4
    ).T

    # Setup plotting components
    (arm13_line,) = ax.plot(
        [], [], [], "r-o", linewidth=3, markersize=8, label="Front-Left / Rear-Right"
    )
    (arm24_line,) = ax.plot(
        [], [], [], "b-o", linewidth=3, markersize=8, label="Front-Right / Rear-Left"
    )
    (trail_line,) = ax.plot([], [], [], "k--", alpha=0.6, label="Flight Path")
    (target_dot,) = ax.plot([], [], [], "go", markersize=10, label="Dynamic Target")

    # Dynamic limits setting
    min_x, max_x = np.min(states[:, 0]) - 1.0, np.max(states[:, 0]) + 1.0
    min_y, max_y = np.min(states[:, 1]) - 1.0, np.max(states[:, 1]) + 1.0
    min_z, max_z = 0.0, np.max(states[:, 2]) + 1.0

    # Make axes uniform bound
    max_range = max(max_x - min_x, max_y - min_y, max_z - min_z) / 2.0
    mid_x = (max_x + min_x) / 2.0
    mid_y = (max_y + min_y) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(0.0, 2.0 * max_range)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Quadcopter 3D Physics Simulation")
    ax.legend()

    # Time display text block
    time_template = "Time = %.1fs\nPosition = (%.2f, %.2f, %.2f)m"
    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def init():
        arm13_line.set_data([], [])
        arm13_line.set_3d_properties([])
        arm24_line.set_data([], [])
        arm24_line.set_3d_properties([])
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        target_dot.set_data([], [])
        target_dot.set_3d_properties([])
        time_text.set_text("")
        return arm13_line, arm24_line, trail_line, target_dot, time_text

    def update_frame(num):
        state = states[num]
        pos = state[0:3]
        phi, theta, psi = state[6:9]

        # Calculate rotation matrix R body coordinates -> world coordinates
        c_ph, s_ph = np.cos(phi), np.sin(phi)
        c_th, s_th = np.cos(theta), np.sin(theta)
        c_ps, s_ps = np.cos(psi), np.sin(psi)

        R = np.array(
            [
                [
                    c_th * c_ps,
                    s_ph * s_th * c_ps - c_ph * s_ps,
                    c_ph * s_th * c_ps + s_ph * s_ps,
                ],
                [
                    c_th * s_ps,
                    s_ph * s_th * s_ps + c_ph * c_ps,
                    c_ph * s_th * s_ps - s_ph * c_ps,
                ],
                [-s_th, s_ph * c_th, c_ph * c_th],
            ]
        )

        # Rotate arm coordinate frames to World frame and shift to modern position
        arm13_world = R @ arm13_body + pos.reshape(3, 1)
        arm24_world = R @ arm24_body + pos.reshape(3, 1)

        # Update lines
        arm13_line.set_data(arm13_world[0, :], arm13_world[1, :])
        arm13_line.set_3d_properties(arm13_world[2, :])

        arm24_line.set_data(arm24_world[0, :], arm24_world[1, :])
        arm24_line.set_3d_properties(arm24_world[2, :])

        # Trailing path line
        trail_line.set_data(states[: num + 1, 0], states[: num + 1, 1])
        trail_line.set_3d_properties(states[: num + 1, 2])

        # Waypoint target marker
        target = targets[num]
        target_dot.set_data([target[0]], [target[1]])
        target_dot.set_3d_properties([target[2]])

        # Telemetry updates text box
        time_text.set_text(time_template % (t[num], pos[0], pos[1], pos[2]))

        return arm13_line, arm24_line, trail_line, target_dot, time_text

    ani = animation.FuncAnimation(
        fig, update_frame, frames=len(states), init_func=init, interval=50, blit=True
    )

    if save_path:
        print(f"Saving animation to file: {save_path}...")
        ani.save(save_path, writer="pillow", fps=20)
        print("Save finished!")
    else:
        plt.show()

    return ani
