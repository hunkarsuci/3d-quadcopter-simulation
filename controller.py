import numpy as np

class PIDController:
    """Standard PID Controller class with anti-windup."""
    def __init__(self, kp, ki, kd, limit_integral=None, limit_output=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit_integral = limit_integral
        self.limit_output = limit_output

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        # Proportional term
        p_term = self.kp * error

        # Integral term with anti-windup clamping
        self.integral += error * dt
        if self.limit_integral is not None:
            self.integral = np.clip(self.integral, -self.limit_integral, self.limit_integral)
        i_term = self.ki * self.integral

        # Derivative term
        d_term = self.kd * (error - self.prev_error) / dt if dt > 0.0 else 0.0
        self.prev_error = error

        output = p_term + i_term + d_term
        if self.limit_output is not None:
            output = np.clip(output, -self.limit_output, self.limit_output)

        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0


class CascadedFlightController:
    """
    Cascaded Flight Controller for a quadcopter.
    Contains position loops (outer) and attitude/rate loops (inner).
    """
    def __init__(self, mass, g, L, c_tf):
        self.mass = mass
        self.g = g
        self.L = L
        self.c_tf = c_tf
        self.l_factor = L / np.sqrt(2.0)

        # 1. Outer Loop: Position Controller (x, y) -> Target Roll/Pitch
        # Output: Desired accelerations or desired Euler angles
        self.pid_x = PIDController(kp=1.5, ki=0.05, kd=2.0, limit_integral=2.0, limit_output=5.0)
        self.pid_y = PIDController(kp=1.5, ki=0.05, kd=2.0, limit_integral=2.0, limit_output=5.0)

        # 2. Altitude Controller: z -> Target Thrust (T)
        # Output: Desired total thrust
        self.pid_z = PIDController(kp=10.0, ki=1.5, kd=8.0, limit_integral=10.0, limit_output=30.0)

        # 3. Inner Loop: Attitude Controller (phi, theta, psi) -> Target angular rates
        # Output: target body rates (p, q, r)
        self.pid_roll  = PIDController(kp=6.0, ki=0.1, kd=1.5, limit_integral=1.0, limit_output=6.0)
        self.pid_pitch = PIDController(kp=6.0, ki=0.1, kd=1.5, limit_integral=1.0, limit_output=6.0)
        self.pid_yaw   = PIDController(kp=4.0, ki=0.05, kd=1.0, limit_integral=0.5, limit_output=3.0)

        # 4. Angular Rate Controller (p, q, r) -> Body Moments (tau_x, tau_y, tau_z)
        # Output: Torques (tau_x, tau_y, tau_z)
        # Often a fast PI or PID loop. Let's make it proportional/integral rate feedback.
        self.pid_p = PIDController(kp=0.5, ki=0.0, kd=0.02, limit_integral=1.0, limit_output=5.0)
        self.pid_q = PIDController(kp=0.5, ki=0.0, kd=0.02, limit_integral=1.0, limit_output=5.0)
        self.pid_r = PIDController(kp=0.8, ki=0.0, kd=0.02, limit_integral=1.0, limit_output=5.0)

        # Constraints
        self.max_tilt = np.deg2rad(25.0) # Maximum commandable roll/pitch angle
        self.max_motor_thrust = 12.0     # Maximum thrust per motor (N)

    def control(self, state, target_pos, target_yaw, dt):
        """
        Runs the cascaded PID control stack.
        - state: dynamics state vector (length 12)
        - target_pos: target 3D coordinates [x, y, z]
        - target_yaw: target yaw angle (rad)
        - dt: timestep duration

        Returns: motor_forces (array of length 4)
        """
        # Unpack state
        x, y, z = state[0:3]
        vx, vy, vz = state[3:6]
        phi, theta, psi = state[6:9]
        p, q, r = state[9:12]

        x_ref, y_ref, z_ref = target_pos

        # --- PID 1: Altitude Control ---
        hover_thrust = self.mass * self.g
        thrust_correction = self.pid_z.update(z_ref - z, dt)
        total_thrust = hover_thrust + thrust_correction

        # --- PID 2: Position Control in World Space ---
        # Compute desired accelerations in world coordinates
        acc_x_des = self.pid_x.update(x_ref - x, dt)
        acc_y_des = self.pid_y.update(y_ref - y, dt)

        # Rotate desired accelerations to the yaw-aligned body frame
        # to find desired roll and pitch
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)
        acc_x_body =  cos_psi * acc_x_des + sin_psi * acc_y_des
        acc_y_body = -sin_psi * acc_x_des + cos_psi * acc_y_des

        # Calculate target Euler angles
        # Acc_x (forward) requires positive pitch theta.
        # Acc_y (leftward, positive y) requires negative roll phi (tilts rightward to slide left? No, let's trace:
        # positive roll tilts rightward, pointing thrust rightward, accelerating rightward (negative y).
        # So leftward acceleration (positive y) requires negative roll.
        theta_des = acc_x_body / self.g
        phi_des = -acc_y_body / self.g

        # Clamp commandable tilt angles
        theta_des = np.clip(theta_des, -self.max_tilt, self.max_tilt)
        phi_des = np.clip(phi_des, -self.max_tilt, self.max_tilt)

        # --- PID 3: Attitude Loop ---
        # Compute desired angular rates (p_des, q_des, r_des)
        p_des = self.pid_roll.update(phi_des - phi, dt)
        q_des = self.pid_pitch.update(theta_des - theta, dt)

        # Yaw error normalization to [-pi, pi]
        yaw_error = target_yaw - psi
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
        r_des = self.pid_yaw.update(yaw_error, dt)

        # --- PID 4: Rate Loop ---
        # Compute target torques
        tau_x = self.pid_p.update(p_des - p, dt)
        tau_y = self.pid_q.update(q_des - q, dt)
        tau_z = self.pid_r.update(r_des - r, dt)

        # --- Motor Mixer ---
        # Map total thrust and torques to the four individual motor forces
        # F1 = 1/4 * (T + tau_x/l_fac - tau_y/l_fac - tau_z/c_tf)
        # F2 = 1/4 * (T - tau_x/l_fac - tau_y/l_fac + tau_z/c_tf)
        # F3 = 1/4 * (T - tau_x/l_fac + tau_y/l_fac - tau_z/c_tf)
        # F4 = 1/4 * (T + tau_x/l_fac + tau_y/l_fac + tau_z/c_tf)
        motor_forces = np.zeros(4)

        mixer_matrix = np.array([
            [1.0,  1.0/self.l_factor, -1.0/self.l_factor, -1.0/self.c_tf],
            [1.0, -1.0/self.l_factor, -1.0/self.l_factor,  1.0/self.c_tf],
            [1.0, -1.0/self.l_factor,  1.0/self.l_factor, -1.0/self.c_tf],
            [1.0,  1.0/self.l_factor,  1.0/self.l_factor,  1.0/self.c_tf]
        ])

        controls = np.array([total_thrust, tau_x, tau_y, tau_z])
        motor_forces = 0.25 * (mixer_matrix @ controls)

        # Clamp motor forces to physical limits [0, max_motor_thrust]
        motor_forces = np.clip(motor_forces, 0.0, self.max_motor_thrust)

        # Return control structure info for logging
        control_log = {
            'throttle': total_thrust,
            'tau_x': tau_x,
            'tau_y': tau_y,
            'tau_z': tau_z,
            'phi_des': phi_des,
            'theta_des': theta_des,
            'p_des': p_des,
            'q_des': q_des,
            'r_des': r_des
        }

        return motor_forces, control_log

    def reset_integrators(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_roll.reset()
        self.pid_pitch.reset()
        self.pid_yaw.reset()
        self.pid_p.reset()
        self.pid_q.reset()
        self.pid_r.reset()
