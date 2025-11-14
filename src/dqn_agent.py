import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    """
    Deep Q-Learning Agent for PID gain tuning.
    """
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995, 
                 buffer_capacity=50000, batch_size=64):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma        
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        self.memory = deque(maxlen=buffer_capacity)
        
        # Build Q-Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear')) 
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        """Copies weights from the main model to the target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Adds a transition to the replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Using the epsilon-greedy strategy - Exploration vs. Exploitation."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.expand_dims(state, axis=0) 
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """
        Trains the model by sampling a batch from the replay buffer.
        Performs the core DQN update using the Bellman equation.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.amax(target_q_values, axis=1)
        
        targets = rewards + self.gamma * max_next_q * (1 - dones)

        current_q_values = self.model.predict(states, verbose=0)
        
        for i in range(self.batch_size):
            current_q_values[i][actions[i]] = targets[i]

        # Training the model to match the updated Q-values
        self.model.fit(states, current_q_values, epochs=1, verbose=0)
        
        # Decay epsilon after replay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)