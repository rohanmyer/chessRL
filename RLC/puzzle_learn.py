from learn import TD_search
import time
import numpy as np
from tqdm import tqdm


class PuzzleLearner(TD_search):
    def __init__(self, env, agent, puzzles):
        super(PuzzleLearner, self).__init__(env, agent)
        self.current_puzzle = None
        self.index = -1
        self.puzzles = puzzles
        self.load_puzzle()

    def load_puzzle(self):
        self.index += 1
        if self.index >= len(self.puzzles):
            self.index = 0
        self.current_puzzle = self.puzzles[self.index]
        self.env.reset()
        self.env.board.set_fen(self.current_puzzle["fen"])
        self.env.init_layer_board()

    def learn(self, iters=40, c=100, timelimit_seconds=3600):
        """
        Start the learning process for the chess agent.
        Args:
            iters: Number of iterations to train for
            c: Frequency of updating the agent model
            timelimit_seconds: Time limit for the training session
            maxiter: Maximum number of moves per game
        """
        starttime = time.time()
        for k in tqdm(range(iters)):
            if k % c == 0:
                self.agent.fix_model()
            if k % 10 == 0:
                self.update_agent()
            self.play_puzzle()
            if starttime + timelimit_seconds < time.time():
                break

    def play_puzzle(self):
        """
        Conduct a single game of chess and learn from it.
        Args:
            maxiter: Maximum number of moves per game
        """
        state = np.expand_dims(self.env.layer_board.copy(), axis=0)
        move = self.select_move(state)
        answer = self.current_puzzle["best_move"]
        episode_end, reward = self.env.puzzle_step(move, answer)
        sucstate = np.expand_dims(self.env.layer_board, axis=0)
        error = self.compute_error(state, reward, sucstate)

        self.record_experience(state, reward, 0, sucstate, error, episode_end)

        self.load_puzzle()
