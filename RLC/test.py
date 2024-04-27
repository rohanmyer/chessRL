import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

import agent
import environment
import learn
import puzzle_learn
import chess
from chess.pgn import Game
import pickle


PUZZLE_PATH = "/Users/rohan/Documents/chessRL/data/lichess_puzzles.pkl"
print("Loading the puzzles...")
with open(PUZZLE_PATH, "rb") as file:
    puzzles = pickle.load(file)
print(f"Number of puzzles: {len(puzzles)}")  # 14,031,988 total

NETWORK = "super_simple"

opponent = agent.GreedyAgent()
env = environment.Board(opponent, FEN=None)
player = agent.Agent(lr=0.01, network=NETWORK)
player.load(f"rlc_{NETWORK}")
learner = learn.TD_search(env, player, gamma=0.8)
# learner = puzzle_learn.PuzzleLearner(env, player, puzzles)
player.model.summary()

learner.learn(iters=20, timelimit_seconds=3600)

player.save(f"rlc_{NETWORK}")

reward_smooth = pd.DataFrame(learner.reward_trace)
reward_smooth.rolling(window=500, min_periods=0).mean().plot(
    figsize=(16, 9), title="average performance"
)
plt.show()

# reward_smooth = pd.DataFrame(learner.piece_balance_trace)
# reward_smooth.rolling(window=100, min_periods=0).mean().plot(
#     figsize=(16, 9), title="average piece balance over the last 3 episodes"
# )
# plt.show()

pgn = Game.from_board(learner.env.board)
with open("rlc_pgn", "w") as log:
    log.write(str(pgn))
