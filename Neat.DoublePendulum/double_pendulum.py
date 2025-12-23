import math
import numpy as np

class DoublePendulum:
    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole1 = 0.1
        self.masspole2 = 0.1
        self.length1 = 0.5 # Half length
        self.length2 = 0.5 # Half length
        # Calculate moments of inertia (assuming thin rods approximation)
        self.I1 = (1.0/12.0) * self.masspole1 * (2*self.length1)**2
        self.I2 = (1.0/12.0) * self.masspole2 * (2*self.length2)**2
        
        self.force_mag = 10.0
        self.tau = 0.02  # Time interval between state updates in seconds
        
        # Thresholds
        # Threshold angle for episode failure (in radians)
        self.theta_threshold_radians = math.pi / 2 # 90 degrees threshold
        self.x_threshold = 2.4

        # State variables: Position, Velocity, Pole 1 Angle, Pole 1 Angular Velocity, Pole 2 Angle, Pole 2 Angular Velocity
        self.state = None
        self.reset()

    def reset(self):
        # Initialize in a horizontal position with slight random noise
        self.state = (
            0.0, 
            0.0, 
            math.pi / 2 + 0.1 * np.random.uniform(-1, 1), 
            0.0, 
            math.pi / 2 + 0.1 * np.random.uniform(-1, 1), 
            0.0
        )
        self.steps_beyond_done = None
        return self.state

    def step(self, action):
        x, x_dot, theta1, theta1_dot, theta2, theta2_dot = self.state
        
        # Determine force direction: 0 for left, 1 for right
        force = self.force_mag if action == 1 else -self.force_mag
        
        # Physics simulation using Lagrangian mechanics
        # Physical constants and state variables
        g = self.gravity
        M = self.masscart
        m1 = self.masspole1
        m2 = self.masspole2
        l1 = self.length1 # Distance to center of mass
        l2 = self.length2 # Distance to center of mass
        L1 = 2 * l1 # Total length of the pole
        
        # Cos/Sin
        c1 = math.cos(theta1)
        s1 = math.sin(theta1)
        c2 = math.cos(theta2)
        s2 = math.sin(theta2)
        c12 = math.cos(theta1 - theta2)
        s12 = math.sin(theta1 - theta2)
        
        # Solve the equation of motion: M(q) * q_ddot = F - C - G
        # q = [x, theta1, theta2]
        
        # Construct the Mass Matrix (M)
        # Row 1 (Eq for x)
        M11 = M + m1 + m2
        M12 = m1*l1*c1 + m2*L1*c1
        M13 = m2*l2*c2
        
        # Row 2 (Eq for theta1)
        M21 = m1*l1*c1 + m2*L1*c1
        M22 = m1*l1**2 + m2*L1**2 + self.I1
        M23 = m2*L1*l2*c12
        
        # Row 3 (Eq for theta2)
        M31 = m2*l2*c2
        M32 = m2*L1*l2*c12
        M33 = m2*l2**2 + self.I2
        
        MassMat = np.array([
            [M11, M12, M13],
            [M21, M22, M23],
            [M31, M32, M33]
        ])
        
        # Construct the Coriolis and Centripetal force vector (C)
        # Centripetal terms dependent on angular velocity squared
        C1 = -(m1*l1*s1 + m2*L1*s1) * theta1_dot**2 - (m2*l2*s2) * theta2_dot**2
        C2 = m2*L1*l2*s12 * theta2_dot**2
        C3 = -m2*L1*l2*s12 * theta1_dot**2
        
        C_vec = np.array([C1, C2, C3])
        
        # Construct the Gravity vector (G)
        G1 = 0
        G2 = -(m1*l1 + m2*L1) * g * s1
        G3 = -m2*l2 * g * s2
        
        G_vec = np.array([G1, G2, G3])
        
        # Define the external force vector
        F_vec = np.array([force, 0, 0])
        
        # Solve for joint accelerations (q_ddot)
        # MassMat * q_ddot = F_vec - C_vec - G_vec
        RHS = F_vec - C_vec - G_vec
        try:
            q_ddot = np.linalg.solve(MassMat, RHS)
        except np.linalg.LinAlgError:
            q_ddot = np.zeros(3)
        
        x_acc = q_ddot[0]
        theta1_acc = q_ddot[1]
        theta2_acc = q_ddot[2]
        
        # Perform Euler integration to update state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        
        theta1 = theta1 + self.tau * theta1_dot
        theta1_dot = theta1_dot + self.tau * theta1_acc
        
        theta2 = theta2 + self.tau * theta2_dot
        theta2_dot = theta2_dot + self.tau * theta2_acc
        
        self.state = (x, x_dot, theta1, theta1_dot, theta2, theta2_dot)
        
        # Episode termination condition: Cart moves beyond track limits
        done = (x < -self.x_threshold) or (x > self.x_threshold)
        
        # Normalize angles for reward calculation
        t1_norm = theta1 % (2*math.pi)
        t2_norm = theta2 % (2*math.pi)
        
        r1 = math.cos(theta1)
        r2 = math.cos(theta2)
        
        dist_penalty = (x/self.x_threshold)**2 # Apply penalty based on distance from the center
        
        reward = (r1 + 1.0) + (r2 + 1.0) - dist_penalty
        
        # Clip reward to be non-negative just in case
        reward = max(0.0, reward)
        
        if done:
            reward = 0.0 # Zero reward on failure
            
        return self.state, reward, done
