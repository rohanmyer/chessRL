import chess
import chess.engine
import gym
import numpy as np

WIN_SCORE = 1000
DRAW_SCORE = 500

ENGINE_LIMIT = chess.engine.Limit(depth=20)

STOCKFISH_PATH = "stockfish"


class ChessEnv(gym.Env):
    def __init__(self):
        self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        self.board = chess.Board()
        self.state = self.board.fen()

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        if type(action) == str:
            move = self.board.parse_uci(action)
        elif action < len(legal_moves):
            move = legal_moves[action]
        else:
            # Handle invalid action (e.g., return an error or a default move)
            print("Invalid action! Using a random action instead.")
            action = np.random.choice(len(legal_moves))
            move = legal_moves[action]

        self.board.push(move)
        self.state = self.board.fen()

        # Get the reward
        score = self.engine.analyse(self.board, ENGINE_LIMIT)["score"]
        reward = score.white().score(mate_score=WIN_SCORE)

        if self.board.is_checkmate():
            reward = WIN_SCORE
            print("checkmate!")
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = DRAW_SCORE
            print("draw!")

        # Get the next state
        self.state = self.board.fen()

        # Check if the game is done
        done = self.board.is_game_over()
        if done:
            return next_state, reward, done, {}

        # Get the engine's move
        result = self.engine.play(self.board, ENGINE_LIMIT)
        self.board.push(result.move)
        self.state = self.board.fen()
        if self.board.is_checkmate():
            reward = -1 * WIN_SCORE
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = DRAW_SCORE
        done = self.board.is_game_over()
        return self.state, reward, done, {}

    def reset(self):
        self.board = chess.Board()
        self.state = self.board.fen()
        return self.state


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
