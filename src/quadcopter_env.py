import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .pid_controller import PIDController

"""
SIMULATION CONSTANTS
G - Gravity (m/s^2) | MASS - Quadcopter mass (kg) | DT - Time step | MAX_STEP - Max steps per RL episode
"""
G = 9.81 
MASS = 1.0 
DT = 0.01 
MAX_STEPS = 100 

""" PID TUNING CONSTANTS """ 
INITIAL_KP = 10.0
INITIAL_KD = 5.0
INITIAL_KI = 0.0 
KP_LIMITS = (0.0, 50.0)
KD_LIMITS = (0.0, 50.0)
STEP_DELTA = 0.5 # This is the magnitude of the discrete gain change action

class QuadcopterPIDA(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.action_space = spaces.Discrete(9)

        # Observation space: (z, dz/dt, error, integral_sum, derivative) + (Kp, Kd)
        # Bounded by a high value (1000) for position/velocity/PID state, and gain limits for Kp/Kd
        high = np.array([
            100.0, # Altitude (z)
            100.0, # Vertical Velocity (dz/dt)
            100.0, # Error (setpoint - z)
            100.0, # Integral Sum (Ki is 0, but included for generality)
            100.0, # Derivative
            KP_LIMITS[1], # Kp value
            KD_LIMITS[1], # Kd value
        ], dtype=np.float32)

        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.pid_controller = PIDController(
            Kp=INITIAL_KP, 
            Ki=INITIAL_KI, 
            Kd=INITIAL_KD, 
            setpoint=1.0 # Setting the target altitude to 1.0m
        )
        
        # Initial quadcopter state (altitude, velocity)
        self.z = 0.0 
        self.z_dot = 0.0
        self.steps_taken = 0
        self.Kp, self.Ki, self.Kd = self.pid_controller.get_gains()


    def _get_obs(self):
        """Returns the current observation vector."""
        error, integral, derivative = self.pid_controller.get_state()
        return np.array([
            self.z,
            self.z_dot,
            error,
            integral,
            derivative,
            self.pid_controller.Kp,
            self.pid_controller.Kd
        ], dtype=np.float32)

    def _get_info(self):
        """Returns diagnostic information."""
        return {
            "Kp": self.pid_controller.Kp,
            "Kd": self.pid_controller.Kd,
            "altitude": self.z,
            "velocity": self.z_dot
        }
        
    def step(self, action):
        """
        Executes one step in the environment. This step consists of:
        1. Applying the RL action (tuning the PID gains).
        2. Running the internal quadcopter simulation for MAX_STEPS.
        3. Calculating the reward based on performance over MAX_STEPS.
        """
        self.steps_taken += 1
        
        # 1. Apply RL Action: Update Kp and Kd based on the discrete action
        d_kp, d_kd = self._map_action_to_gain_change(action)
        
        new_Kp = np.clip(self.pid_controller.Kp + d_kp, *KP_LIMITS)
        new_Kd = np.clip(self.pid_controller.Kd + d_kd, *KD_LIMITS)
        
        self.pid_controller.set_gains(new_Kp, self.pid_controller.Ki, new_Kd)
        
        # 2. Run internal simulation for a short duration (MAX_STEPS * DT)
        total_squared_error = 0.0
        
        # --- FIX 1: Initialize altitude history array ---
        altitude_history = [] 
        
        for _ in range(MAX_STEPS):
            # PID generates control force
            control_force = self.pid_controller.update(self.z, DT)
            
            F_thrust = np.clip(control_force, 0.0, 2.0 * MASS * G) # max thrust is 2x gravity
            
            F_net = F_thrust - (MASS * G)
            acceleration = F_net / MASS
            
            # Euler integration for dynamics
            self.z_dot += acceleration * DT
            self.z += self.z_dot * DT
            
            # --- FIX 2: Record altitude at every physics step ---
            altitude_history.append(self.z)
            
            # Accumulate Integrated Squared Error (ISE) for reward calculation
            error = self.pid_controller.setpoint - self.z
            total_squared_error += error**2
            
            # Check for catastrophic failure (e.g., drone crashed through the floor or flew too high)
            if self.z < -0.5 or self.z > 5.0:
                 terminated = True
                 break

        # 3. Calculate Reward: Negative ISE, and a terminal penalty if unstable
        # The goal is to minimize ISE 
        reward = -total_squared_error 
        
        terminated = self.steps_taken >= 100 # Terminate after 100 RL steps (100 * 1s = 100s total sim time)
        truncated = False # Not using explicit truncation for now

        # Large penalty for crashing/flying away
        if self.z < -0.5 or self.z > 5.0:
            reward -= 1000.0 
            terminated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        # --- FIX 3: Add altitude history to the info dictionary ---
        info['altitude_history'] = altitude_history

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets the environment and quadcopter to initial conditions."""
        super().reset(seed=seed)
        
        # Reset physical state slightly above ground
        self.z = self.np_random.uniform(low=0.1, high=0.3) 
        self.z_dot = 0.0
        self.steps_taken = 0
        
        # Keep PID gains from the last episode (this is the essence of tuning)
        # Only reset the integral/error state of the PID controller
        self.pid_controller.reset() 

        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def _map_action_to_gain_change(self, action):
        """Maps a discrete action index (0-8) to (dKp, dKd)."""
        # 0: +Delta, 1: 0, 2: -Delta
        dKp_map = {0: STEP_DELTA, 1: STEP_DELTA, 2: STEP_DELTA, 
                   3: 0.0, 4: 0.0, 5: 0.0, 
                   6: -STEP_DELTA, 7: -STEP_DELTA, 8: -STEP_DELTA}
                   
        dKd_map = {0: STEP_DELTA, 1: 0.0, 2: -STEP_DELTA, 
                   3: STEP_DELTA, 4: 0.0, 5: -STEP_DELTA, 
                   6: STEP_DELTA, 7: 0.0, 8: -STEP_DELTA}

        dKp_index = action // 3
        dKd_index = action % 3

        # Align the 9 indices with the action table:
        if action == 0: return STEP_DELTA, STEP_DELTA
        if action == 1: return STEP_DELTA, 0.0
        if action == 2: return STEP_DELTA, -STEP_DELTA
        
        if action == 3: return 0.0, STEP_DELTA
        if action == 4: return 0.0, 0.0
        if action == 5: return 0.0, -STEP_DELTA
        
        if action == 6: return -STEP_DELTA, STEP_DELTA
        if action == 7: return -STEP_DELTA, 0.0
        if action == 8: return -STEP_DELTA, -STEP_DELTA
        
        return 0.0, 0.0 # Should not happen

    def render(self):
        print(f"Time: {self.steps_taken*MAX_STEPS*DT:.2f}s | Z: {self.z:.2f}m | Kp: {self.pid_controller.Kp:.2f}, Kd: {self.pid_controller.Kd:.2f}")

    def close(self):
        pass