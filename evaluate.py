from rl_chess import agent, environment, learn

from tqdm import tqdm

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

puzzle_player = agent.Agent(lr=0.01, network="custom")
puzzle_player.load("rl_chess/weights/may_1_11hr/puzzle_custom")
players.append({"player": puzzle_player, "name": "puzzle_base", "engine": False})

self_player = agent.Agent(lr=0.01, network="custom")
self_player.load("rl_chess/weights/may_1_11hr/self_custom")
players.append({"player": self_player, "name": "self_base", "engine": False})

ft_puzzle_player = agent.Agent(lr=0.01, network="custom")
ft_puzzle_player.load("rl_chess/weights/self_custom")
players.append({"player": ft_puzzle_player, "name": "puzzle_ft", "engine": False})

ft_self_player = agent.Agent(lr=0.01, network="custom")
ft_self_player.load("rl_chess/weights/puzzle_custom")
players.append({"player": ft_self_player, "name": "self_ft", "engine": False})

weak_engine = agent.EngineAgent(ENGINE_PATH, depth=1)
players.append({"player": weak_engine, "name": "stockfish_1", "engine": True})
medium_engine = agent.EngineAgent(ENGINE_PATH, depth=10)
players.append({"player": medium_engine, "name": "stockfish_10", "engine": True})
strong_engine = agent.EngineAgent(ENGINE_PATH, depth=20)
players.append({"player": strong_engine, "name": "stockfish_20", "engine": True})

results = {}
for white_player in players:
    for black_player in players:
        white_wins = 0
        black_wins = 0
        for i in range(100):
            env = environment.Board(FEN=None)
            if white_player["engine"]:
                white_model = white_player["player"]
            else:
                white_model = learn.TD_search(
                    env, white_player["player"], name=white_player["name"], epsilon=-1.0
                )
            if black_player["engine"]:
                black_model = black_player["player"]
            else:
                black_model = learn.TD_search(
                    env, black_player["player"], name=black_player["name"], epsilon=-1.0
                )

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
        results[(white_player["name"], black_player["name"])] = (white_wins, black_wins)

import json

with open("results.json", "w") as file:
    json.dump(results, file)
