from rl_chess import agent, environment, learn
import os
import pickle
import sys

WEIGHTS_PATH = "rl_chess/weights"
ENGINE_PATH = "/users/rkrish16/data/rkrish16/other/chessRL/data/stockfish/stockfish-ubuntu-x86-64-avx2"
NETWORK = "big"
env = environment.Board(FEN=None)
player = agent.Agent(lr=0.01, network=NETWORK)


def self_train():
    if os.path.exists(f"{WEIGHTS_PATH}/self_{NETWORK}"):
        player.load(f"{WEIGHTS_PATH}/self_{NETWORK}")
    learner = learn.TD_search(env, player, gamma=0.8)
    player.model.summary()

    learner.learn(iters=int(1e6), timelimit_seconds=3600)


def puzzle_train():
    PUZZLE_PATH = "data/lichess_puzzles.pkl"
    print("Loading the puzzles...")
    with open(PUZZLE_PATH, "rb") as file:
        puzzles = pickle.load(file)
    print(f"Number of puzzles: {len(puzzles)}")  # 14,031,988 total

    if os.path.exists(f"{WEIGHTS_PATH}/puzzle_{NETWORK}"):
        player.load(f"{WEIGHTS_PATH}/puzzle_{NETWORK}")
    learner = learn.PuzzleLearner(env, player, puzzles)
    player.model.summary()

    learner.learn(iters=int(1e6), timelimit_seconds=3600)


def engine_train():
    opponent = agent.EngineAgent(ENGINE_PATH)
    name = f"{opponent.color}_engine"
    if os.path.exists(f"{WEIGHTS_PATH}/{name}_{NETWORK}"):
        player.load(f"{WEIGHTS_PATH}/{name}_{NETWORK}")
    learner = learn.EngineLearner(env, player, opponent)
    player.model.summary()

    learner.learn(iters=int(1e6), timelimit_seconds=3600)


if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "self":
        self_train()
    elif mode == "puzzle":
        puzzle_train()
    elif mode == "engine":
        engine_train()
    else:
        print("Invalid")
