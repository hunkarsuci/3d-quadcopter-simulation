import os
import numpy as np

# Ensure execution works if run from outer parent directories
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from .quadcopter import Quadcopter
    from .controller import CascadedFlightController
    from .simulator import QuadcopterSimulator
except ImportError:
    from quadcopter import Quadcopter
    from controller import CascadedFlightController
    from simulator import QuadcopterSimulator

def main():
    try:
        from .animate import plot_flight_telemetry, animate_copter_3d
    except ImportError:
        from animate import plot_flight_telemetry, animate_copter_3d

    # 1. Initialize physical Drone plant
    quad = Quadcopter(
        mass=1.5,          # Kg
        Ixx=0.08,          # Kg*m^2
        Iyy=0.08,          # Kg*m^2
        Izz=0.15,          # Kg*m^2
        L=0.25,            # arm length (meters)
        c_tf=0.015,        # thrust-to-drag-torque coefficient
        c_drag=0.1         # air resistance coefficient
    )

    # 2. Setup cascaded controller matching the physical parameters
    controller = CascadedFlightController(
        mass=quad.mass,
        g=quad.g,
        L=quad.L,
        c_tf=quad.c_tf
    )

    # 3. Create simulation model
    sim = QuadcopterSimulator(quad, controller)

    # 4. Set mission waypoints
    # Format: (duration_seconds, [x_target, y_target, z_target], yaw_deg_target)
    # The drone will execute the flight trajectory one waypoint at a time.
    waypoints = [
        (3.0,  [0.0, 0.0, 1.0],   0.0),    # Take-off to 1m altitude
        (3.0,  [1.0, 0.0, 1.0],  45.0),    # Travel to X = 1.0m, rotate yaw to 45 deg
        (3.0,  [1.0, 1.0, 1.5],  90.0),    # Travel to X=1m, Y=1m, climb to 1.5m, rotate yaw to 90 deg
        (3.0,  [0.0, 1.0, 1.2], 180.0),    # Travel to X=0m, Y=1m, hover at 1.2m, face opposite side (180 deg)
        (3.0,  [0.0, 0.0, 1.0],   0.0),    # Return to origin hover, reset yaw to 0 deg
        (3.0,  [0.0, 0.0, 0.0],   0.0),    # Command descent to ground level
    ]
    sim.set_waypoints(waypoints)

    # 5. Run simulation
    # Total time = sum of all waypoints durations (3s * 6 = 18s). Let's simulate for 18.0s.
    history = sim.run(total_time=18.0, dt=0.002, control_frequency=250)

    # 6. Prompt visualization options
    # We will generate static telemetry graphs and save the 3D flight animation.
    print("\nGenerating static flight telemetries...")
    plot_flight_telemetry(history)

    # Export a visual GIF animation of the flight space
    gif_filename = os.path.join(current_dir, "drone_flight.gif")
    animate_copter_3d(history, arm_length=quad.L, save_path=gif_filename)
    print(f"\nCompleted! 3D animation saved as: {gif_filename}")

if __name__ == '__main__':
    main()
