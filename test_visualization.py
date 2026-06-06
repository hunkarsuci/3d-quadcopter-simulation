import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from animate import animate_copter_3d, plot_flight_telemetry
from controller import CascadedFlightController
from quadcopter import Quadcopter
from simulator import QuadcopterSimulator


class TestVisualizationSmoke(unittest.TestCase):
    """Headless smoke tests for Matplotlib plotting and animation."""

    def _short_history(self):
        quad = Quadcopter()
        controller = CascadedFlightController(quad.mass, quad.g, quad.L, quad.c_tf)
        sim = QuadcopterSimulator(quad, controller)
        sim.set_waypoints([(0.08, [0.0, 0.0, 0.2], 0.0)])
        return sim.run(total_time=0.08, dt=0.004, control_frequency=125)

    def test_telemetry_plot_headless(self):
        history = self._short_history()
        fig = plot_flight_telemetry(history, show=False)
        self.assertIsNotNone(fig)
        self.assertGreater(len(plt.get_fignums()), 0)
        plt.close("all")

    def test_animation_gif_headless(self):
        history = self._short_history()

        with tempfile.TemporaryDirectory() as tmpdir:
            gif_path = os.path.join(tmpdir, "flight.gif")
            animate_copter_3d(history, save_path=gif_path)

            self.assertTrue(os.path.exists(gif_path))
            self.assertGreater(os.path.getsize(gif_path), 0)

        plt.close("all")


if __name__ == "__main__":
    unittest.main()
