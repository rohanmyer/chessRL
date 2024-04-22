from chess_env import ChessPuzzleEnv
from q_learning_agent import QLearningAgent

import numpy as np

import pickle

# Load the puzzles
with open("../data/lichess_puzzles.pkl", "rb") as file:
    puzzles = pickle.load(file)

# Create the environment and agent
env = ChessPuzzleEnv(puzzles)
agent = QLearningAgent()

# Train the agent
for episode in range(1000):
    state = env.reset()
    done = False
    rewards = 0

    legal_moves = list(env.board.legal_moves)
    action = agent.act(state, legal_moves)
    next_state, reward, done, _ = env.step(action)
    agent.update(state, legal_moves, action, next_state, reward)
    state = next_state
    rewards += reward
    print(f"Episode {episode+1}, Reward: {rewards}")
    print("State: \n", env.board)

# Save the Q-values
np.save("q_values.npy", agent.q_values)

# Load the Q-values
# agent.q_values = np.load("q_values.npy", allow_pickle=True).item()

# Close the engine
env.engine.quit()