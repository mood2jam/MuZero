from checkers import *
from muzero_agent import *
import argparse
import multiprocessing

# COMM = MPI.COMM_WORLD
# RANK = COMM.Get_rank()
# RANK = 0
parser = argparse.ArgumentParser()
parser.add_argument('--num_games', type=int, help='True or False, whether or not to train network', default=100)
parser.add_argument('--num_workers', type=int, help='True or False, whether or not to train network', default=1)
args = parser.parse_args()

# Initialize our muzero agent
def play_game(RANK):
	agent = MuZero_Agent(agent_num = RANK, device="cuda:0")
	num_games = args.num_games
	if RANK == 0:
		agent.game_num = 500
	elif RANK == 1:
		agent.game_num = 122	
	print("Process {} started.".format(RANK))
	for i in range(num_games):
		print("Agent {0} started game {1}".format(RANK, agent.game_num))
		game = Game("computer1", "computer2", show_graphics=False, print_board=False, delay=0, agent1=agent, agent2=agent)
		game.play() # Have the agent play against itself
		print("Agent {0} finished game {1}".format(RANK, agent.game_num))
		agent.game_num += 1
		agent.game_data = None

	print("Process {} ended.".format(RANK))


if __name__ == "__main__":
    game_threads = []
    for worker in range(args.num_workers):
        t = multiprocessing.Process(target=play_game, args=(worker,))
        game_threads.append(t)

    for thread in game_threads:
        thread.start()

    for thread in game_threads:
        thread.join()

    print("Done!")
    play_game(0)


