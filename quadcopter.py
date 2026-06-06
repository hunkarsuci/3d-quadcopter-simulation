import numpy as np


class Quadcopter:
    """
    Quadcopter 3D Physics and Dynamics Model.

    Coordinate System (Aeronautical conventions: Front-Left-Up):
    - X-axis (+): Front
    - Y-axis (+): Left
    - Z-axis (+): Up

    Motor Layout (X-Configuration):
    - Motor 1: Front-Left (FL), CCW (Reacts CW -> Negative Yaw Torque)
    - Motor 2: Front-Right (FR), CW (Reacts CCW -> Positive Yaw Torque)
    - Motor 3: Rear-Right (RR), CCW (Reacts CW -> Negative Yaw Torque)
    - Motor 4: Rear-Left (RL), CW (Reacts CCW -> Positive Yaw Torque)
    """

    def __init__(
        self,
        mass=1.5,  # Kg
        Ixx=0.08,  # Kg*m^2
        Iyy=0.08,  # Kg*m^2
        Izz=0.15,  # Kg*m^2
        L=0.25,  # Arm length (m)
        c_tf=0.015,  # Drag torque to thrust ratio (m)
        c_drag=0.1,  # Translational drag coefficient
        g=9.81,
    ):  # Gravity (m/s^2)

        self.mass = mass
        self.I = np.diag([Ixx, Iyy, Izz])
        self.I_inv = np.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz])
        self.L = L
        self.c_tf = c_tf
        self.c_drag = c_drag
        self.g = g

        # State vector: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        # pos = state[0:3]
        # vel = state[3:6]
        # attitude (euler) = state[6:9]
        # angular velocity (omega) = state[9:12]
        self.state = np.zeros(12)

        # Initialize position above ground slightly
        self.state[2] = 0.0

    @property
    def position(self):
        return self.state[0:3]

    @position.setter
    def position(self, val):
        self.state[0:3] = val

    @property
    def velocity(self):
        return self.state[3:6]

    @velocity.setter
    def velocity(self, val):
        self.state[3:6] = val

    @property
    def euler(self):
        return self.state[6:9]

    @euler.setter
    def euler(self, val):
        self.state[6:9] = val

    @property
    def omega(self):
        return self.state[9:12]

    @omega.setter
    def omega(self, val):
        self.state[9:12] = val

    def get_rotation_matrix(self):
        """
        Returns rotation matrix R from Body Frame to World Frame.
        Uses Z-Y-X Euler angle convention: R = Rz(psi) * Ry(theta) * Rx(phi)
        """
        phi, theta, psi = self.euler

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
        return R

    def compute_derivatives(self, state, motor_forces):
        """
        Computes the state derivative dx/dt for a given state and motor forces.
        """
        # Unpack state
        # pos = state[0:3]
        vel = state[3:6]
        euler = state[6:9]
        omega = state[9:12]

        phi, theta, psi = euler
        p, q, r = omega

        # Calculate total thrust (force in body Z axis direction)
        total_thrust = np.sum(motor_forces)

        # Calculate forces & torques
        # Motor mapping to torques (X-configuration):
        # T_roll (tau_x)  = L/sqrt(2) * (F1 - F2 - F3 + F4)
        # T_pitch (tau_y) = L/sqrt(2) * (-F1 - F2 + F3 + F4)
        # T_yaw (tau_z)   = c_tf * (-F1 + F2 - F3 + F4)
        F1, F2, F3, F4 = motor_forces
        l_factor = self.L / np.sqrt(2.0)

        tau_x = l_factor * (F1 - F2 - F3 + F4)
        tau_y = l_factor * (-F1 - F2 + F3 + F4)
        tau_z = self.c_tf * (-F1 + F2 - F3 + F4)

        torques = np.array([tau_x, tau_y, tau_z])

        # Rotation Matrix
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

        # 1. Translational equations: mdv/dt = m*g + R*F_thrust - c_drag*v
        g_vec = np.array([0.0, 0.0, -self.g])
        thrust_body = np.array([0.0, 0.0, total_thrust])
        thrust_world = R @ thrust_body

        # Drag acceleration
        accel_drag = -(self.c_drag / self.mass) * vel

        accel = g_vec + (thrust_world / self.mass) + accel_drag

        # 2. Rotational attitude rates: deuler/dt = W * omega
        # Singular if theta ~= pi/2, but safe for normal flight
        cos_theta = np.cos(theta)
        if abs(cos_theta) < 1e-4:
            cos_theta = 1e-4 * np.sign(cos_theta)

        W = np.array(
            [
                [1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                [0.0, np.cos(phi), -np.sin(phi)],
                [0.0, np.sin(phi) / cos_theta, np.cos(phi) / cos_theta],
            ]
        )
        euler_dot = W @ omega

        # 3. Rotational dynamics: I * domega/dt = tau - omega x (I * omega)
        omega_dot = self.I_inv @ (torques - np.cross(omega, self.I @ omega))

        # Assemble dx/dt
        dstatedt = np.zeros(12)
        dstatedt[0:3] = vel
        dstatedt[3:6] = accel
        dstatedt[6:9] = euler_dot
        dstatedt[9:12] = omega_dot

        return dstatedt

    def step(self, motor_forces, dt):
        """
        Integrates dynamic equations by one time-step using 4th-order Runge-Kutta.
        """
        # Clamp motor forces to physical limits (non-negative)
        motor_forces = np.clip(motor_forces, 0.0, None)

        # RK4 Integration
        k1 = self.compute_derivatives(self.state, motor_forces)
        k2 = self.compute_derivatives(self.state + 0.5 * dt * k1, motor_forces)
        k3 = self.compute_derivatives(self.state + 0.5 * dt * k2, motor_forces)
        k4 = self.compute_derivatives(self.state + dt * k3, motor_forces)

        self.state += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Simple ground constraint (cannot fly below ground)
        if self.state[2] < 0.0:
            self.state[2] = 0.0  # Position z = 0
            self.state[5] = 0.0  # Velocity vz = 0
            # Damp lateral movement under contact
            self.state[3:5] *= 0.5
            self.state[9:12] *= 0.5  # Damp angular rate on crash/ground
            self.state[6:8] *= 0.5  # Heading remains, roll/pitch flatten
