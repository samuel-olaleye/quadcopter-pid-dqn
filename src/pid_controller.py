import numpy as np

class PIDController:
    
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, integral_limit=1.0):
        
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint

        self.last_error = 0.0
        self.integral_sum = 0.0
        self.integral_limit = integral_limit

        # RL observation 
        self.last_proportional_error = 0.0
        self.last_derivative_error = 0.0


    def update(self, current_value, dt):
        
        # 1. Calculate Error
        error = self.setpoint - current_value

        # 2. Proportional Term
        P_term = self.Kp * error

        # 3. Integral Term 
        self.integral_sum += error * dt
        if self.integral_sum > self.integral_limit:
            self.integral_sum = self.integral_limit
        elif self.integral_sum < -self.integral_limit:
            self.integral_sum = -self.integral_limit

        I_term = self.Ki * self.integral_sum

        # 4. Derivative Term (Handle dt=0 safely)
        if dt > 0:
            derivative = (error - self.last_error) / dt
            D_term = self.Kd * derivative
        else:
            D_term = 0.0 

        # 5. Total PID Output
        output = P_term + I_term + D_term

        # Update internal state for the next step
        self.last_error = error
        self.last_proportional_error = error 
        self.last_derivative_error = derivative 

        return output

    def reset(self):
        self.last_error = 0.0
        self.integral_sum = 0.0
        self.last_proportional_error = 0.0
        self.last_derivative_error = 0.0

    def get_gains(self):
        return self.Kp, self.Ki, self.Kd

    def set_gains(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
    def get_state(self):
        return (self.last_proportional_error, self.integral_sum, self.last_derivative_error)