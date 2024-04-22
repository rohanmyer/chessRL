import numpy as np


class QLearningAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, gamma=0.9, user_play=False):
        self.q_values = {}
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q_values = {}
        self.user_play = user_play

    def act(self, state, legal_moves):
        if self.user_play:
            print("State: ")
            print(state)
            print("Your Move: ")
            move = input()
            return move

        # Choose an action (move) using epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(legal_moves))
        else:
            action = np.argmax(self.q_values.get(state, [0] * len(legal_moves)))
        return action

    def update(self, state, legal_moves, action, next_state, reward):
        # Update Q-values using Q-learning update rule
        q_value = self.q_values.get(state, [0] * len(legal_moves))[action]
        next_q_value = self.q_values.get(next_state, [0] * len(legal_moves))[action]
        self.q_values[state] = self.q_values.get(state, [0] * len(legal_moves))
        self.q_values[state][action] = q_value + self.alpha * (
            reward + self.gamma * next_q_value - q_value
        )
