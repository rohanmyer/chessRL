from rl_chess import agent, environment, learn

from tqdm import tqdm


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


puzzle_player = agent.Agent(lr=0.01, network="big")
puzzle_player.load("rl_chess/weights/puzzle_big")

self_player = agent.Agent(lr=0.01, network="big")
self_player.load("rl_chess/weights/self_big")

engine_player = agent.Agent(lr=0.01, network="big")
engine_player.load("rl_chess/weights/black_engine_big")

white_player = self_player
black_player = self_player

n = 10

white_wins = 0
black_wins = 0

for i in range(n):
    env = environment.Board(FEN=None)
    white_model = learn.TD_search(env, white_player, name="puzzle")
    black_model = learn.TD_search(env, black_player, name="greedy")
    print(f"Game {i+1}")
    while env.board.result() == "*":
        move = white_model.select_move(None)
        env.board.push(move)
        env.update_layer_board(move)

        if env.board.result() == "1-0":
            print(f"White win after {env.board.fullmove_number} moves")
            white_wins += 1
            break
        elif env.board.result() == "1/2-1/2":
            print(f"White draw after {env.board.fullmove_number} moves")
            print(draw_reason(env.board))
            break

        move = black_model.select_move(None)
        env.board.push(move)
        env.update_layer_board(move)

        if env.board.result() == "1-0":
            print(f"Black win after {env.board.fullmove_number} moves")
            black_wins += 1
            break
        elif env.board.result() == "1/2-1/2":
            print(f"Black draw after {env.board.fullmove_number} moves")
            print(draw_reason(env.board))
            break
    print()

print(f"White wins: {white_wins}")
print(f"Black wins: {black_wins}")
