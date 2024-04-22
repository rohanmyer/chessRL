from chess_env import ChessEnv
from q_learning_agent import QLearningAgent

import numpy as np

# Create the environment and agent
env = ChessEnv()
agent = QLearningAgent()

# Train the agent
for episode in range(1000):
    state = env.reset()
    done = False
    rewards = 0
    count = 0
    while not done:
        count += 1
        legal_moves = list(env.board.legal_moves)
        action = agent.act(state, legal_moves)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, legal_moves, action, next_state, reward)
        state = next_state
        rewards += reward
        if count % 20 == 0:
            print(f"Episode {episode+1}, Reward: {rewards}, Moves: {count}")
            print("State: \n", env.board)
    print(f"Episode {episode+1}, Reward: {rewards}, Moves: {count}")
    print("State: \n", env.board)

# Save the Q-values
np.save("q_values.npy", agent.q_values)

# Load the Q-values
# agent.q_values = np.load("q_values.npy", allow_pickle=True).item()

# Close the engine
env.engine.quit()
