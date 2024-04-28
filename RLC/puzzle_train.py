import agent
import environment
import learn

import pickle


PUZZLE_PATH = "/users/rkrish16/data/rkrish16/other/chessRL/data/lichess_puzzles.pkl"
print("Loading the puzzles...")
with open(PUZZLE_PATH, "rb") as file:
    puzzles = pickle.load(file)
print(f"Number of puzzles: {len(puzzles)}")  # 14,031,988 total

NETWORK = "big"

env = environment.Board(FEN=None)
player = agent.Agent(lr=0.01, network=NETWORK)
player.load(f"puzzle_{NETWORK}")
learner = learn.PuzzleLearner(env, player, puzzles)
player.model.summary()

learner.learn(iters=int(1e6), timelimit_seconds=3600)
