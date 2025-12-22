import math
import random

class CartPole:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 1.0 # Represents half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # Time interval between state updates in seconds
        self.kinematics_integrator = 'euler'

        # The angle threshold (in radians) at which the episode fails
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # State variables: cart position (x), cart velocity (x_dot), pole angle (theta), and pole angular velocity (theta_dot)
        self.state = None
        self.reset()

    def reset(self):
        # Initialize with a random state close to zero, with a slight tilt
        self.state = (random.uniform(-0.05, 0.05), 0, random.uniform(-0.05, 0.05), 0)
        self.steps_beyond_done = None
        return self.state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        
        # The action determines direction: 0 for left, 1 for right
        force = self.force_mag if action == 1 else -self.force_mag
        
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = (x < -self.x_threshold) or (x > self.x_threshold) or \
               (theta < -self.theta_threshold_radians) or (theta > self.theta_threshold_radians)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # The pole has fallen below the threshold
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                pass
            self.steps_beyond_done += 1
            reward = 0.0

        return self.state, reward, done
