from rl_chess import agent, environment, learn
import os
import pickle
import sys

WEIGHTS_PATH = "/users/rkrish16/data/rkrish16/other/chessRL/rl_chess/weights"
ENGINE_PATH = "/users/rkrish16/data/rkrish16/other/chessRL/data/stockfish/stockfish-ubuntu-x86-64-avx2"
NETWORK = "custom"
env = environment.Board(FEN=None)
player = agent.Agent(lr=0.001, network=NETWORK)


def self_train():
    PUZZLE_PATH = "/users/rkrish16/data/rkrish16/other/chessRL/data/lichess_puzzles.pkl"
    print("Loading the puzzles...")
    with open(PUZZLE_PATH, "rb") as file:
        puzzles = pickle.load(file)
    print(f"Number of puzzles: {len(puzzles)}")  # 14,031,988 total

    player.load(
        f"/users/rkrish16/data/rkrish16/other/chessRL/rl_chess/weights/valid/self_custom"
    )
    learner = learn.PuzzleLearner(env, player, puzzles, name="self_ft")
    player.model.summary()

    learner.learn(iters=int(1e6), timelimit_seconds=3600)


def puzzle_self():
    player.load(
        f"/users/rkrish16/data/rkrish16/other/chessRL/rl_chess/weights/puzzle_custom"
    )
    learner = learn.TD_search(env, player, gamma=0.8, name="puzzle_self")
    player.model.summary()

    learner.learn(iters=int(1e6), timelimit_seconds=3600)


def puzzle_engine():
    opponent = agent.EngineAgent(ENGINE_PATH, depth=1)
    player.load(
        f"/users/rkrish16/data/rkrish16/other/chessRL/rl_chess/weights/puzzle_custom"
    )
    learner = learn.EngineLearner(env, player, opponent, name="puzzle_engine")
    player.model.summary()

    learner.learn(iters=int(1e6), timelimit_seconds=3600)


def engine_train():
    PUZZLE_PATH = "/users/rkrish16/data/rkrish16/other/chessRL/data/lichess_puzzles.pkl"
    print("Loading the puzzles...")
    with open(PUZZLE_PATH, "rb") as file:
        puzzles = pickle.load(file)
    print(f"Number of puzzles: {len(puzzles)}")  # 14,031,988 total
    player.load(
        f"/users/rkrish16/data/rkrish16/other/chessRL/rl_chess/weights/valid/black_engine_custom"
    )
    learner = learn.PuzzleLearner(env, player, puzzles, name="engine_ft")
    player.model.summary()

    learner.learn(iters=int(1e6), timelimit_seconds=3600)


if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "self":
        self_train()
    elif mode == "puzzle_self":
        puzzle_self()
    elif mode == "puzzle_engine":
        puzzle_engine()
    elif mode == "engine":
        engine_train()
    else:
        print("Invalid")
