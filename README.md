# Quadrotor PID Auto-Tuner using Deep Q-Networks (DQN)

## Project Overview

This project implements a Deep Q-Network (DQN) agent to automatically discover optimal Proportional ($K_p$) and Derivative ($K_d$) gains for a simplified quadrotor (quadcopter) altitude controller. The goal is to use Reinforcement Learning (RL) to achieve stable and efficient altitude tracking, overcoming the traditional trial-and-error process of manual PID tuning.

---

## Technology Stack

- **Reinforcement Learning:** Deep Q-Networks (DQN)  
- **Frameworks:** TensorFlow / Keras, NumPy, Pandas  
- **Simulation:** Custom Gymnasium-compatible environment (`src/quadcopter_env.py`)  
- **Notebook:** `pid_dqn_training.ipynb` for training, analysis and visualization

---

## Setup and Execution

### Prerequisites

Ensure you have Python (3.8+) and the required packages installed in a virtual environment.

```bash
# Assuming you are in the project root directory:
python -m venv .venv

# Activate the environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## Training the Agent

Training is executed entirely within the Jupyter notebook **`pid_dqn_training.ipynb`**.  
Run all cells sequentially to ensure correct path setup, training, and analysis.

---

## Results and Performance Analysis

The DQN agent was trained over **500 episodes**. The following visualizations demonstrate the agent's learning process and the performance of the final, optimized PID controller.

### 1. Learning Curve (Reward vs. Episode)

This plot tracks the average total reward (Negative Integrated Squared Error) over 10-episode windows.
![DQN Performance (Total Reward per Episode) Graph](image-1.png)

**Analysis:**  
The training demonstrates clear convergence, indicating successful learning. The average reward (error magnitude) dropped steeply from initial values of approximately **−4200** (high instability) to stabilize around **−1200**. This **70% reduction** in error magnitude demonstrates the DQN agent effectively minimized the altitude tracking error by discovering superior PID gains compared to the initial random settings.

---

### 2. Learned Gain Trajectory (Kp and Kd Evolution)

This plot shows how the values of **Kp** and **Kd** evolved throughout the 500 episodes as the agent explored the gain space.
![DQN Auto-Tuning: Trajectory of PID Gain Graph](image-2.png)

**Analysis:**  
- **Kp** converged close to **1.0**  
- **Kd** converged around **7.5**

This suggests the optimal controller for this simulated environment relies primarily on **strong damping (Kd)** with **modest proportional feedback (Kp)**.

---

### 3. Final Performance Simulation

The final test runs the quadcopter with the optimized gains learned by the DQN agent (e.g., **Kp ≈ 1.0, Kd ≈ 7.5**) and plots the altitude (z-axis) over time.

**Summary:**  
The optimized PID controller, tuned by the DQN agent, successfully stabilized the quadcopter's altitude at the target (**1.0 m**) with:
- Minimal overshoot  
- Fast settling time (≈ **1.5 seconds**)  
- Smooth, stable response

This validates the Deep Reinforcement Learning approach as a viable method for automating control system tuning.

---

## Project Structure

├── .venv/ # Python Virtual Environment
├── models/ # Saved Keras weights (.h5 file)
├── results/ # Training logs and generated plots
├── src/
│ ├── dqn_agent.py # Deep Q-Network model and logic
│ ├── quadcopter_env.py # Custom Gymnasium environment
│ ├── pid_controller.py # Basic PID controller implementation
│ └── train.py # Main script for executing the training loop
├── pid_dqn_training.ipynb # Main runnable notebook
└── requirements.txt