from rl_chess import agent, environment, learn

from tqdm import tqdm

import gc

ENGINE_PATH = "/users/rkrish16/data/rkrish16/other/chessRL/data/stockfish/stockfish-ubuntu-x86-64-avx2"


def draw_reason(board):
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    if board.is_seventyfive_moves():
        return "75 moves"
    if board.is_fivefold_repetition():
        return "fivefold repetition"
    if board.is_variant_draw():
        return "variant draw"
    return "unknown"


players = []

players.append(
    {
        "player": "rl_chess/weights/may_1_11hr/puzzle_custom",
        "name": "puzzle_base",
        "engine": False,
    }
)

players.append(
    {
        "player": "rl_chess/weights/may_1_11hr/self_custom",
        "name": "self_base",
        "engine": False,
    }
)

players.append(
    {"player": "rl_chess/weights/self_custom", "name": "puzzle_ft", "engine": False}
)

players.append(
    {"player": "rl_chess/weights/puzzle_custom", "name": "self_ft", "engine": False}
)

weak_engine = agent.EngineAgent(ENGINE_PATH, depth=1)
players.append({"player": weak_engine, "name": "stockfish_1", "engine": True})
medium_engine = agent.EngineAgent(ENGINE_PATH, depth=10)
players.append({"player": medium_engine, "name": "stockfish_10", "engine": True})
strong_engine = agent.EngineAgent(ENGINE_PATH, depth=20)
players.append({"player": strong_engine, "name": "stockfish_20", "engine": True})

results = {}
env = environment.Board(FEN=None)
for white_player in players:
    if white_player["engine"]:
        white = white_player["player"]
    else:
        white = agent.Agent(lr=0.1, network="custom")
        white.load(white_player["player"])
    for black_player in players:
        white_wins = 0
        black_wins = 0
        if black_player["engine"]:
            black = black_player["player"]
        else:
            black = agent.Agent(lr=0.1, network="custom")
            black.load(black_player["player"])

        for i in range(10):
            env.reset()
            if white_player["engine"]:
                white_model = white
            else:
                white_model = learn.TD_search(env, white, name=white, epsilon=-1.0)
            if black_player["engine"]:
                black_model = black
            else:
                black_model = learn.TD_search(env, black, name=black, epsilon=-1.0)

            print(f"Game {i+1}")
            while env.board.result() == "*":
                if white_player["engine"]:
                    move = white_model.predict(env.board)
                else:
                    move = white_model.select_move(None)
                env.board.push(move)
                env.update_layer_board(move)

                if env.board.is_checkmate():
                    print(f"White win after {env.board.fullmove_number} moves")
                    white_wins += 1
                    break
                elif env.board.result() == "1/2-1/2":
                    print(f"White draw after {env.board.fullmove_number} moves")
                    print(draw_reason(env.board))
                    break

                if black_player["engine"]:
                    move = black_model.predict(env.board)
                else:
                    move = black_model.select_move(None)
                env.board.push(move)
                env.update_layer_board(move)

                if env.board.is_checkmate():
                    print(f"Black win after {env.board.fullmove_number} moves")
                    black_wins += 1
                    break
                elif env.board.result() == "1/2-1/2":
                    print(f"Black draw after {env.board.fullmove_number} moves")
                    print(draw_reason(env.board))
                    break
            print()

        print(f"{white_player['name']} vs {black_player['name']}")
        print(f"White wins: {white_wins}")
        print(f"Black wins: {black_wins}")
        print()
        if results[white_player["name"]]:
            results[white_player["name"]][black_player["name"]] = (
                white_wins,
                black_wins,
            )
        else:
            results[white_player["name"]] = {
                black_player["name"]: (white_wins, black_wins)
            }
        gc.collect()

import json

with open("results.json", "w") as file:
    json.dump(results, file)
