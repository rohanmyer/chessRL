from chess_env import ChessPuzzleEnv
from q_learning_agent import QLearningAgent

import numpy as np

import pickle

COUNT_PER_SAVE = 1000000

# Load the puzzles
print("Loading the puzzles...")
with open("data/lichess_puzzles.pkl", "rb") as file:
    puzzles = pickle.load(file)
print(f"Number of puzzles: {len(puzzles)}") # 14,031,988

# Create the environment and agent
env = ChessPuzzleEnv(puzzles)
agent = QLearningAgent()

# Train the agent
print("Training the agent...")
from tqdm import tqdm
pbar = tqdm(range(COUNT_PER_SAVE))
# for episode in range(1000):
episode = 0
ep_count = 0
correct = 0
while True:
    state = env.reset()
    done = False
    rewards = 0

    legal_moves = list(env.board.legal_moves)
    action = agent.act(state, legal_moves)
    next_state, reward, done, _ = env.step(action)
    agent.update(state, legal_moves, action, next_state, reward)
    state = next_state
    rewards += reward
    episode += 1
    ep_count += 1
    pbar.update(1)
    if reward > 0:
        correct += 1

    if episode % COUNT_PER_SAVE == 0:
        print("Current episode: ", episode)
        acc = 100 * correct/ep_count
        print(f"Accuracy in this {COUNT_PER_SAVE}: ", acc)
        acc = int(acc*100)
        print("Saving the Q-values...")
        np.save(f"weights/puzzle_q_values_{int(episode/COUNT_PER_SAVE)}_{acc}.npy", agent.q_values)
        pbar.refresh()
        pbar.reset()
        correct = 0
        ep_count = 0

# Save the Q-values
np.save("q_values.npy", agent.q_values)

# Load the Q-values
# agent.q_values = np.load("q_values.npy", allow_pickle=True).item()

# Close the engine
env.e