import chess
import chess.engine
import numpy as np

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5


class Board(object):

    def __init__(self, FEN=None, capture_reward_factor=0.01):
        """
        Chess Board Environment
        Args:
            FEN: str
                Starting FEN notation, if None then start in the default chess position
            capture_reward_factor: float [0,inf]
                reward for capturing a piece. Multiply material gain by this number. 0 for normal chess.
        """
        self.FEN = FEN
        self.capture_reward_factor = capture_reward_factor
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.layer_board = np.zeros(shape=(8, 8, 8))
        self.init_layer_board()

        self.engine = chess.engine.SimpleEngine.popen_uci(
            "/users/rkrish16/data/rkrish16/other/chessRL/data/stockfish/stockfish-ubuntu-x86-64-avx2"
        )
        self.limit = chess.engine.Limit(depth=10)
        self.win_score = 10000

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:

        """
        self.layer_board = np.zeros(shape=(8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[layer, row, col] = sign
            self.layer_board[6, :, :] = 1 / self.board.fullmove_number
        if self.board.turn:
            self.layer_board[6, 0, :] = 1
        else:
            self.layer_board[6, 0, :] = -1
        self.layer_board[7, :, :] = 1

    def update_layer_board(self, move=None):
        self._prev_layer_board = self.layer_board.copy()
        self.init_layer_board()

    def pop_layer_board(self):
        self.layer_board = self._prev_layer_board.copy()
        self._prev_layer_board = None

    def step(self, action, test=True):
        """
        Run a step
        Args:
            action: python chess move
        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: float
                Difference in material value after the move
        """
        piece_balance_before = self.get_material_value()
        player = self.board.turn
        self.board.push(action)
        self.update_layer_board(action)
        piece_balance_after = self.get_material_value()
        auxiliary_reward = (
            piece_balance_after - piece_balance_before
        ) * self.capture_reward_factor
        score = self.engine.analyse(self.board, self.limit)["score"]
        if player == chess.WHITE:
            reward = score.white().score(mate_score=self.win_score)
        else:
            reward = score.black().score(mate_score=self.win_score)
        result = self.board.result()
        if result == "*":
            # reward = 0
            episode_end = False
        elif result == "1-0":
            # reward = 1
            episode_end = True
        elif result == "0-1":
            # reward = -1
            episode_end = True
        elif result == "1/2-1/2":
            # reward = 0
            episode_end = True
        # reward += auxiliary_reward

        return episode_end, reward

    def puzzle_step(self, action, answer):
        # piece_balance_before = self.get_material_value()
        self.board.push(action)
        self.update_layer_board(action)
        # piece_balance_after = self.get_material_value()
        # auxiliary_reward = (
        #     piece_balance_after - piece_balance_before
        # ) * self.capture_reward_factor
        if self.board.is_checkmate():
            reward = self.win_score
            episode_end = True
        elif action.uci() == answer:
            reward = self.win_score
            episode_end = True
        else:
            reward = -1 * self.win_score
            episode_end = True
        # reward += auxiliary_reward

        return episode_end, reward

    def _black_engine_step(self, action, engine):
        self.board.push(action)
        self.update_layer_board(action)
        score = self.engine.analyse(self.board, self.limit)["score"]
        reward = score.white().score(mate_score=self.win_score)
        if self.board.is_checkmate():
            episode_end = True
        else:
            engine_move = engine.predict(self.board)
            self.board.push(engine_move)
            self.update_layer_board(engine_move)
            score = self.engine.analyse(self.board, self.limit)["score"]
            if self.board.is_checkmate():
                # reward = score.white().score(mate_score=self.win_score)
                episode_end = True
            else:
                # reward = score.white().score(mate_score=self.win_score)
                episode_end = False
        return episode_end, reward

    def _white_engine_step(self, action, engine):
        engine_move = engine.predict(self.board)
        self.board.push(engine_move)
        self.update_layer_board(engine_move)
        piece_balance_after = self.get_material_value()
        if self.board.is_checkmate():
            score = self.engine.analyse(self.board, self.limit)["score"]
            reward = score.black().score(mate_score=self.win_score)
            episode_end = True
        else:
            self.board.push(action)
            self.update_layer_board(action)
            score = self.engine.analyse(self.board, self.limit)["score"]
            reward = score.black().score(mate_score=self.win_score)
            if self.board.is_checkmate():
                episode_end = True
            else:
                episode_end = False

        return episode_end, reward

    def engine_step(self, action, engine):
        if engine.color == "white":
            return self._white_engine_step(action, engine)
        else:
            return self._black_engine_step(action, engine)

    def get_random_action(self):
        """
        Sample a random action
        Returns: move
            A legal python chess move.

        """
        legal_moves = [x for x in self.board.generate_legal_moves()]
        legal_moves = np.random.choice(legal_moves)
        return legal_moves

    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        self.action_space = np.zeros(shape=(64, 64))
        moves = [
            [x.from_square, x.to_square] for x in self.board.generate_legal_moves()
        ]
        for move in moves:
            self.action_space[move[0], move[1]] = 1
        return self.action_space

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * np.sum(self.layer_board[0, :, :])
        rooks = 5 * np.sum(self.layer_board[1, :, :])
        minor = 3 * np.sum(self.layer_board[2:4, :, :])
        queen = 9 * np.sum(self.layer_board[4, :, :])
        return pawns + rooks + minor + queen

    def reset(self):
        """
        Reset the environment
        Returns:

        """
        self.board = chess.Board(self.FEN) if self.FEN else chess.Board()
        self.init_layer_board()
