import numpy as np
import time
import math
import gc
from tqdm import tqdm


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TD_search(object):
    def __init__(self, env, agent, gamma=0.9, memsize=2000, batch_size=256):
        """
        Initialize a chess algorithm that uses Q Learning for decision making.
        Args:
            env: Chess environment
            agent: Chess playing agent
            gamma: Discount factor for future rewards
            memsize: Size of the memory buffer for storing experiences
            batch_size: Number of experiences to sample from memory during training
        """
        self.env = env
        self.agent = agent
        self.gamma = gamma
        self.memsize = memsize
        self.batch_size = batch_size
        self.mem_state = np.zeros(shape=(1, 8, 8, 8))
        self.mem_sucstate = np.zeros(shape=(1, 8, 8, 8))
        self.mem_reward = np.zeros(shape=(1))
        self.mem_error = np.zeros(shape=(1))
        self.mem_episode_active = np.ones(shape=(1))
        self.reward_trace = []
        self.piece_balance_trace = []

    def learn(self, iters=40, c=5, timelimit_seconds=3600, maxiter=80):
        """
        Start the learning process for the chess agent.
        Args:
            iters: Number of iterations to train for
            c: Frequency of updating the agent model
            timelimit_seconds: Time limit for the training session
            maxiter: Maximum number of moves per game
        """
        starttime = time.time()
        for k in range(iters):
            self.env.reset()
            if k % c == 0:
                self.agent.fix_model()
                print("Iteration", k)
            self.play_game(maxiter)
            if starttime + timelimit_seconds < time.time():
                break

    def play_game(self, maxiter):
        """
        Conduct a single game of chess and learn from it.
        Args:
            k: Current game iteration number
            maxiter: Maximum number of half-moves allowed in a game
        """
        episode_end = False
        turn_count = 0

        pbar = tqdm(total=maxiter, desc="Moves in game")
        while not episode_end and turn_count < maxiter:
            state = np.expand_dims(self.env.layer_board.copy(), axis=0)
            move = self.select_move(state)
            episode_end, reward = self.env.step(move)
            sucstate = np.expand_dims(self.env.layer_board, axis=0)
            error = self.compute_error(state, reward, sucstate)
            piece_balance = 0  # self.env.piece_balance()

            self.record_experience(
                state, reward, piece_balance, sucstate, error, episode_end
            )
            turn_count += 1
            if turn_count % 10 == 0:
                self.update_agent()
            pbar.update(1)

        print(f"Game ended with reward {reward} in {turn_count} half-moves.")

    def select_move(self, state):
        """
        Select a move based on the current state of the board.
        """
        move_values = {}
        for move in self.env.board.generate_legal_moves():
            self.env.step(move)
            next_state = np.expand_dims(self.env.layer_board, axis=0)
            move_values[move] = self.agent.predict(next_state)
            self.env.board.pop()
        best_move = max(move_values, key=move_values.get)
        return best_move

    def compute_error(self, state, reward, sucstate):
        """
        Compute the error for a transition.
        """
        state_value = self.agent.predict(state)
        new_state_value = self.agent.predict(sucstate)
        return reward + self.gamma * new_state_value - state_value

    def record_experience(
        self, state, reward, piece_balance, sucstate, error, episode_end
    ):
        """
        Record experience in the replay buffer.
        """
        self.mem_state = np.append(self.mem_state, state, axis=0)
        self.mem_reward = np.append(self.mem_reward, reward)
        self.mem_sucstate = np.append(self.mem_sucstate, sucstate, axis=0)
        self.mem_error = np.append(self.mem_error, error)
        self.reward_trace = np.append(self.reward_trace, reward)
        self.piece_balance_trace = np.append(self.piece_balance_trace, piece_balance)
        self.mem_episode_active = np.append(
            self.mem_episode_active, 1 if not episode_end else 0
        )
        if self.mem_state.shape[0] > self.memsize:
            self.reduce_memory()

    def update_agent(self):
        """
        Update the agent based on stored experiences.
        """
        if self.mem_state.shape[0] >= self.batch_size:
            indices = np.random.choice(
                range(len(self.mem_state)), size=self.batch_size, replace=False
            )
            batch_states = self.mem_state[indices]
            batch_rewards = self.mem_reward[indices]
            batch_sucstates = self.mem_sucstate[indices]
            batch_episode_active = self.mem_episode_active[indices]
            td_errors = self.agent.TD_update(
                batch_states,
                batch_rewards,
                batch_sucstates,
                batch_episode_active,
                self.gamma,
            )
            for idx, error in zip(indices, td_errors):
                self.mem_error[idx] = error

    def reduce_memory(self):
        """
        Reduce the size of memory buffers to enforce memory constraints.
        """
        self.mem_state = self.mem_state[1:]
        self.mem_reward = self.mem_reward[1:]
        self.mem_sucstate = self.mem_sucstate[1:]
        self.mem_error = self.mem_error[1:]
        self.mem_episode_active = self.mem_episode_active[1:]
        gc.collect()
