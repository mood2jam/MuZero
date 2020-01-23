from networks import Repr_Net, Predict_Net, Dynamics_Net
import torch
from torch.distributions import Categorical
import pickle
import random
import numpy as np

from train import get_latest_weights, get_latest_networks, get_a_rep


class MuZero_Agent:
	def __init__(self, agent_num = 0, save_data = True, strategy="stochastic", temperature=1., device = 'cpu', weights = None, get_latest_weights=True):
		"""
		Args
			agent_num (int): An identifier for the agent
			save_data (bool): Whether or not we want to save the game data after we are finished playing
			strategy (str): Either 'stochastic' (used in self play) or 'deterministic' (used when evaluating)
			device (str): Device we want to run our nueral networks on
			temperature (float between 0 and 1): A lower value makes the network take more greedy actions in stochastic mode
			weights (list of .pt objects): weights to be loaded in the current nueral networks
		"""

		if weights is not None:
			# Load weights here
			pass
		if get_latest_weights:
			print("Got Latest Weights")
			self.repr_net, self.dynamics_net, self.predict_net = get_latest_networks()
		else:
			self.repr_net = Repr_Net()
			self.predict_net = Predict_Net()
			self.dynamics_net = Dynamics_Net()

		self.repr_net = self.repr_net.to(device)
		self.predict_net = self.predict_net.to(device)
		self.dynamics_net = self.dynamics_net.to(device)

		self.strategy = strategy
		self.temperature = temperature
		self.c1 = 1.25
		self.c2 = 19652
		self.gamma = 1
		self.num_sims = 50
		self.num_actions = 512
		self.game_data = torch.zeros((1, 4 + self.num_actions + 17*8*8))
		self.states_visited = [None]
		self.device = device
		self.num_unroll_steps = 5
		self.save_data = save_data
		self.agent_num = agent_num
		self.game_num = 0
		self.observations_per_state = 4
		self.state_buffer = []

	def trans_player(self, player_rep):
		"""
		Custom function depending on game we are playing
		"""
		BLUE = (0, 0, 255)
		if player_rep == BLUE:
			return 0
		else:
			return 1

	def trans_state(self, state_reps, player):
		"""
		Custom function that translates the state to a tensor depending on the game we are playing

		Params
			A list of states taken from the game
		"""
		state = torch.zeros((16, 8, 8))
		# the default shape for state_reps is 4 x 8 x 8
		for k, state_rep in enumerate(state_reps):
			for i in range(8):
				for j in range(8):
					if state_rep[i][j] == 'B-':
						state[0 + 2 * k, i, j] = 1.
					elif state_rep[i][j] == 'BK':
						state[1 + 2 * k, i, j] = 1.
					elif state_rep[i][j] == 'R-':
						state[0 + 2 * (k + self.observations_per_state), i, j] = 1.
					elif state_rep[i][j] == 'RK':
						state[1 + 2 * (k + self.observations_per_state), i, j] = 1.

		if player == 1:
			state = torch.cat([state, torch.ones((1, 8, 8))], dim=0)
		else:
			state = torch.cat([state, torch.zeros((1, 8, 8))], dim=0)

		# This translates into 17 x 8 x 8 state that we can feed into our networks
		return state

	def run_MCTS(self, s0, masked_actions): # 88% of the time
		"""
		Monte Carlo Tree Search with help from the nueral networks

		Args
			s0 (tensor): the representation of our state after we have run it through the representation function
			masked_actions (ndarray): A mask of possible actions we can take from s0
		"""
		l = self.num_actions

		# Initialize data structures
		self.state_trans = torch.ones((1,l))*float('inf')
		self.Q, self.R, self.N, self.reward = torch.zeros((1,l)).to(self.device), torch.zeros((1,l)).to(self.device), torch.zeros((1,l)).to(self.device), torch.zeros((1,l)).to(self.device)
		self.P, _ = self.predict_net(s0) # s0 is a list of all starting states from all games
		self.get_s_rep = [s0]
		self.path = list()
		self.num_states = 1

		# Run a certain number of simulations
		for i in range(self.num_sims):
			s = 0
			while True:
				new_s, a = self.select(s, torch.from_numpy(masked_actions)) # 12% of the time
				self.path.append((s, a))
				if new_s == float('inf'):
					policy, value = self.expand(s, a) # 65% of the time <- this is the one to speed up
					self.backup(policy, value) # 5% of the time
					break
				else:
					s = int(new_s)

		return self.N[0] # Return the true MCTS policy

	def select(self, s, masked_actions):
		conf_bound = self.Q[s] + self.P[s]*(torch.sqrt(torch.sum(self.N[s]))/(1+self.N[s]))*(self.c1+torch.log(torch.sum((self.N[s]) + self.c2 + 1)/self.c2))
		if s == 0:
			# We will only know the legal actions at the starting state
			if len(torch.nonzero(conf_bound)) == 0:
				a = random.choice(torch.nonzero(masked_actions)).item()
				return self.state_trans[s, a].item(), a

			conf_bound[~masked_actions.to(bool)] = float('-inf')

		a = torch.argmax(conf_bound).item()
		return self.state_trans[s, a].item(), a

	def expand(self, s, a):
		s_rep = self.get_s_rep[s]
		a_rep = get_a_rep(a).to(self.device)

		new_s_rep, reward = self.dynamics_net(torch.cat([s_rep, a_rep], dim=1))
		policy, value = self.predict_net(new_s_rep) # Get a policy and value for our new state
		self.get_s_rep.append(new_s_rep)
		self.num_states += 1

		# Initialize new values for our node
		self.N = torch.cat([self.N, torch.zeros((1,self.num_actions)).to(self.device)], dim=0)
		self.Q = torch.cat([self.Q, torch.zeros((1,self.num_actions)).to(self.device)], dim=0)
		self.P = torch.cat([self.P, policy], dim=0)
		self.state_trans = torch.cat([self.state_trans, torch.ones((1,self.num_actions))*float('inf')], dim=0)
		self.reward = torch.cat([self.reward, torch.zeros((1,self.num_actions)).to(self.device)], dim=0)

		self.state_trans[s, a] = self.num_states - 1 # Keep track of how to transition to the state we just visited
		self.reward[s, a] = reward.item()

		return policy, value

	def backup(self, policy, value):
		discounted = value
		for s, a in reversed(self.path):
			G = discounted
			self.Q[s, a] = (self.N[s, a]*self.Q[s, a] + G) / (self.N[s, a] + 1)
			self.N[s, a] += 1
			G += self.reward[s, a] + self.gamma*discounted
		self.path = []


	def take_action(self, raw_state, masked_actions, player_rep, reward=0, return_=0):
		# Translate the state and player into something we can work with
		player = self.trans_player(player_rep)
		state_rep = self.trans_state(raw_state, player)

		# Get the state and policy
		s0 = self.repr_net(state_rep.to(self.device).unsqueeze(0)) 	# Input a bunch of states into this depending on how many games we are playing at once
		N0 = self.run_MCTS(s0, masked_actions) 											# This will use one set of nueral networks no matter how many games we are looking at

		if self.strategy == "deterministic":
			N0 = N0.to(torch.float32)
			P0 = N0/torch.sum(N0)
			action = torch.argmax(N0).item()
		else:
			N0 = N0.to(torch.float32)
			P0 = N0**(1/self.temperature)/torch.sum(N0**(1/self.temperature))
			m = Categorical(P0)
			action = m.sample().item()

		self.states_visited.append(state_rep)

		# Formatted as player, action, return, reward, policy, state
		assert (state_rep.shape[0] - 1) // 4 == self.observations_per_state
		assert state_rep.shape[1] == 8
		assert state_rep.shape[2] == 8

		# Assert everything is in the right shape
		state_info = torch.cat([torch.tensor([player, action, return_, reward]).to(torch.float32).to(self.device), P0, state_rep.reshape(state_rep.shape[0]*state_rep.shape[1]*state_rep.shape[2]).to(self.device)], dim=0)

		if self.game_data is None:
			self.game_data = state_info.unsqueeze(0)
		else:
			self.game_data = torch.cat([self.game_data.to(self.device), state_info.unsqueeze(0)], dim=0)

		return action

	def save_game_data(self, winner):
		# Updates the return for the winner

		if winner == 0:
			self.game_data[:, 2] = self.game_data[:, 0]*(-2) + 1
		elif winner == 1:
			self.game_data[:, 2] = self.game_data[:, 0]*2 - 1

		shape = self.game_data.shape
		indices = self.game_data.nonzero().cpu()
		values = self.game_data[indices[:,0], indices[:,1]].cpu()

		with open('./game_data/game_{0}_agent_{1}.pickle'.format(self.game_num, self.agent_num), 'wb') as f:
			pickle.dump([shape, indices, values], f, protocol=pickle.HIGHEST_PROTOCOL)

	def evaluate(self, game_state, masked_actions, player_turn, reward=0, winner=None):
		if winner is not None:
			winner = self.trans_player(winner)
			if self.save_data:
				self.save_game_data(winner)
			self.game_data = None
			action = None
		else:
			self.state_buffer.append(game_state)
			if len(self.state_buffer) > self.observations_per_state:
				self.state_buffer.pop(0)
			action = self.take_action(self.state_buffer[::-1], masked_actions, player_turn)

		return action





















