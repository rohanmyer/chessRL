import chess
import chess.engine
import gym
import numpy as np

WIN_SCORE = 1000
DRAW_SCORE = 500

ENGINE_LIMIT = chess.engine.Limit(depth=20)

# STOCKFISH_PATH = "/oscar/data/csun45/rkrish16/other/chessRL/data/stockfish/"
STOCKFISH_PATH = (
    "/Users/rohan/Documents/chessRL/data/stockfish/stockfish-macos-m1-apple-silicon"
)


class ChessEnv(gym.Env):
    def __init__(self, load_engine=True):
        if load_engine:
            self.engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        else:
            self.engine = None
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
            # print("checkmate!")
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = DRAW_SCORE
            # print("draw!")

        # Get the next state
        self.state = self.board.fen()

        # Check if the game is done
        done = self.board.is_game_over()
        if done:
            return self.state, reward, done, {}

        # Get the engine's move
        result = self._engine_move(self.state)
        self.board.push(result.move)
        self.state = self.board.fen()
        if self.board.is_checkmate():
            reward = -1 * WIN_SCORE
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward = DRAW_SCORE
        done = self.board.is_game_over()
        return self.state, reward, done, {}

    def _engine_move(self, fen_state):
        return self.engine.play(self.board, ENGINE_LIMIT)

    def reset(self):
        self.board = chess.Board()
        self.state = self.board.fen()
        return self.state


class ChessPuzzleEnv(ChessEnv):
    def __init__(self, puzzles):
        """
        Args:
            puzzles (list): List of puzzles in the form of a dictionary with keys 'fen' and 'best_move'
        """
        super().__init__(load_engine=False)
        self.current_puzzle = None
        self.index = -1
        self.puzzles = puzzles
        self.load_puzzle()

    def load_puzzle(self):
        self.index += 1
        if self.index >= len(self.puzzles):
            self.index = 0
        self.current_puzzle = self.puzzles[self.index]
        self.board.set_fen(self.current_puzzle["fen"])
        self.state = self.board.fen()

    def step(self, action):
        legal_moves = list(self.board.legal_moves)
        move = legal_moves[action]
        self.board.push(move)

        # Get the reward
        if self.board.is_checkmate():
            reward = WIN_SCORE
        elif move.uci() == self.current_puzzle["best_move"]:
            reward = WIN_SCORE * 0.75
        else:
            reward = -1 * DRAW_SCORE

        self.reset()

        return self.state, reward, True, {}

    def reset(self):
        self.load_puzzle()
        return self.board.fen()
