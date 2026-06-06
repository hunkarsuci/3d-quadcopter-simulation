import unittest
import numpy as np
import sys
import os

# Adjust paths to import the simulation package correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone_simulation.quadcopter import Quadcopter
from drone_simulation.controller import CascadedFlightController
from drone_simulation.simulator import QuadcopterSimulator


class TestDroneSimulationSmoke(unittest.TestCase):
    """Basic smoke tests checking physics and controller convergence."""

    def test_hover_convergence(self):
        """Checks that the drone converges to a commanded target altitude."""
        quad = Quadcopter(mass=1.5, g=9.81)
        controller = CascadedFlightController(quad.mass, quad.g, quad.L, quad.c_tf)
        sim = QuadcopterSimulator(quad, controller)

        # Command to rise to 2.0 meters and hover
        sim.set_waypoints([
            (8.0, [0.0, 0.0, 2.0], 0.0)
        ])

        # Run simulation for 8.0 seconds
        dt = 0.002
        history = sim.run(total_time=8.0, dt=dt, control_frequency=250)

        # Get final state
        final_state = history['state'][-1]
        final_position = final_state[0:3]
        final_velocity = final_state[3:6]
        final_euler = final_state[6:9]

        # assertions
        # 1. Drone should reach target altitude (z ~ 2.0m) within a tolerance (e.g. 10cm)
        self.assertAlmostEqual(final_position[2], 2.0, delta=0.1,
                               msg=f"Drone did not converge to reference altitude: {final_position[2]}m")

        # 2. X and Y positions should remain close to 0
        self.assertAlmostEqual(final_position[0], 0.0, delta=0.02)
        self.assertAlmostEqual(final_position[1], 0.0, delta=0.02)

        # 3. Terminal velocity should be near zero (stationary hover)
        self.assertAlmostEqual(np.linalg.norm(final_velocity), 0.0, delta=0.05)

        # 4. Terminal attitude should be flat
        self.assertAlmostEqual(np.linalg.norm(final_euler), 0.0, delta=0.02)

    def test_ballistic_fall(self):
        """Verifies physics engine correctness: zero thrust yields downward gravity acceleration."""
        quad = Quadcopter(mass=1.5, g=9.81)
        # Drop drone with zero thrust
        zero_forces = np.zeros(4)
        dt = 0.01

        # Preset initial position to z = 10m
        quad.state[2] = 10.0

        # Run dynamics state transitions
        for _ in range(10):
            quad.step(zero_forces, dt)

        # Expected velocity after 0.1s: v_z = -g * t = -9.81 * 0.1 = -0.981 m/s
        # Z position should decrease
        self.assertTrue(quad.state[2] < 10.0)
        self.assertAlmostEqual(quad.state[5], -0.981, delta=0.05)


if __name__ == '__main__':
    unittest.main()
