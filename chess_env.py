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
            return self.state, reward, done, {}

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
