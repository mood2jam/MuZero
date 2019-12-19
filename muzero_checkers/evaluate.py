from checkers import *
from muzero_agent import *
import argparse
import multiprocessing
import numpy as np

BLUE     = (  0,   0, 255)
RED      = (255,   0,   0)

# COMM = MPI.COMM_WORLD
# RANK = COMM.Get_rank()
# RANK = 0
parser = argparse.ArgumentParser()
parser.add_argument('--num_games', type=int, help='True or False, whether or not to train network', default=100)
parser.add_argument('--num_workers', type=int, help='True or False, whether or not to train network', default=1)
args = parser.parse_args()

# Initialize our muzero agent
def evaluate_against_random(RANK, agent, evaluation_num):
  num_games = args.num_games
  wins_as_starting_player = np.zeros(num_games//2)
  wins_as_second_player = np.zeros(num_games//2)
  total_moves_as_starting_player = np.zeros(num_games//2)
  total_moves_as_second_player = np.zeros(num_games//2)
  print("Process {} started.".format(RANK))
  for i in range(num_games//2):
    print("Agent {0} started game {1}".format(RANK, agent.game_num))
    game = Game("random_computer", "computer", show_graphics=False, print_board=False, delay=0, agent2=agent)
    winner, num_moves1 = game.play() # Have the agent play against itself
    print("Number of moves:", num_moves1)
    print("Agent {0} finished game {1}".format(RANK, agent.game_num))
    agent.game_num += 1
    if winner == RED:
      print("Agent won.")
      wins_as_second_player[i] = 1
    total_moves_as_starting_player[i] = num_moves1
  for i in range(num_games // 2):
    print("Agent {0} started game {1}".format(RANK, agent.game_num))
    game = Game("computer", "random_computer", show_graphics=False, print_board=False, delay=0, agent1=agent)
    winner, num_moves2 = game.play()
    print(num_moves2)
    print("Agent {0} finished game {1}".format(RANK, agent.game_num))
    agent.game_num += 1
    if winner == BLUE:
      wins_as_starting_player[i] = 1
    total_moves_as_second_player[i] = num_moves2

  np.save("./evaluations/evaluation_against_random_{0}.npy".format(evaluation_num), np.vstack([wins_as_starting_player, total_moves_as_starting_player, wins_as_second_player, total_moves_as_second_player]))

  print("Process {} ended.".format(RANK))


if __name__ == "__main__":
    game_threads = []
    random_agent = MuZero_Agent(agent_num=0, device="cpu:0", save_data = False)
    for worker in range(args.num_workers):
        t = multiprocessing.Process(target=evaluate_against_random, args=(worker, random_agent, 1,))
        game_threads.append(t)

    for thread in game_threads:
        thread.start()

    for thread in game_threads:
        thread.join()

    print("Done!")


