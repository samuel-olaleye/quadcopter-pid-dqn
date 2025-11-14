import numpy as np
import gymnasium as gym
import os
import time
import pandas as pd
from .quadcopter_env import QuadcopterPIDA
from .dqn_agent import DQNAgent

"""Training Configuration"""
TOTAL_EPISODES = 500      
TARGET_UPDATE_FREQ = 10   
SAVE_FREQ = 100           
# CRITICAL FIX: Use simple paths relative to the project root.
LOG_FILENAME = os.path.join('results', 'training_log.csv')
MODEL_FILENAME = os.path.join('models', 'dqn_pid_tuner.weights.h5')


def train_dqn_pid_tuner():
    # Environment and Agent Setup
    env = QuadcopterPIDA()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(
        state_size=state_size, 
        action_size=action_size,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9997, 
        buffer_capacity=100000,
        batch_size=64
    )

    log_data = []
    print("--- Starting DQN Training for PID Auto-Tuning ---")
    
    # Training Loop
    start_time = time.time()
    
    for episode in range(1, TOTAL_EPISODES + 1):
        state, _ = env.reset() 
        state = np.array(state) 
        total_reward = 0
        step_count = 0
        
        initial_Kp, initial_Ki, initial_Kd = env.pid_controller.get_gains()
        
        while True:
            action = agent.act(state)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                break

        agent.replay()
        
        final_Kp, final_Ki, final_Kd = env.pid_controller.get_gains()
        
        log_data.append({
            'episode': episode,
            'total_reward': total_reward,
            'steps': step_count,
            'epsilon': agent.epsilon,
            'initial_Kp': initial_Kp,
            'final_Kp': final_Kp,
            'initial_Kd': initial_Kd,
            'final_Kd': final_Kd
        })
        
        # Checkpoint and Logic update
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
            
        if episode % SAVE_FREQ == 0:
            agent.save(MODEL_FILENAME)

        # Episode summary
        if episode % 10 == 0:
            avg_reward = np.mean([d['total_reward'] for d in log_data[-10:]])
            print(f"Ep: {episode:4d}/{TOTAL_EPISODES} | Reward: {total_reward:9.2f} | Avg R (10): {avg_reward:7.2f} | Kp/Kd: {final_Kp:.2f}/{final_Kd:.2f} | Epsilon: {agent.epsilon:.4f}")

    agent.save(MODEL_FILENAME)
    df = pd.DataFrame(log_data)
    df.to_csv(LOG_FILENAME, index=False)
    
    end_time = time.time()

    print(f"\n--- Training Finished ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Final model saved to: {MODEL_FILENAME}")
    print(f"Training log saved to: {LOG_FILENAME}")

if __name__ == '__main__':
    try:
        train_dqn_pid_tuner()
    except Exception as e:
        print(f"An error occurred during training: {e}")