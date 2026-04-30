import numpy as np

class QLearningAgent:
    def __init__(self, state_bins=[10, 10, 10, 5, 5]):
        self.state_bins = state_bins
        self.q_table = {}
        self.alpha = 0.1  # скорость обучения
        self.gamma = 0.9  # коэффициент дисконтирования
        self.epsilon = 0.1  # исследование vs использование
        
    def discretize_state(self, state):
        # Дискретизируем непрерывное состояние
        bounds = [
            (-50, 50),     # dino_y
            (-15, 15),     # dino_vy
            (0, 600),      # distance
            (0, 60),       # height
            (0, 50)        # width
        ]
        
        discrete = []
        for i, val in enumerate(state):
            low, high = bounds[i]
            bins = self.state_bins[i]
            if val < low: val = low
            if val > high: val = high
            bin_idx = int((val - low) / (high - low) * (bins - 1))
            discrete.append(min(bin_idx, bins - 1))
            
        return tuple(discrete)
    
    def get_action(self, state):
        state_key = self.discretize_state(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(2)
            
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state_key])
    
    def learn(self, state, action, reward, next_state):
        state_key = self.discretize_state(state)
        next_state_key = self.discretize_state(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(2)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(2)
            
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_key][action] = new_q