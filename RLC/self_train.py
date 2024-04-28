import agent
import environment
import learn

NETWORK = "big"

env = environment.Board(FEN=None)
player = agent.Agent(lr=0.01, network=NETWORK)
player.load(f"game_{NETWORK}")
learner = learn.TD_search(env, player, gamma=0.8)
player.model.summary()

learner.learn(iters=int(1e6), timelimit_seconds=3600)
