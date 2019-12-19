import pickle
import torch
import os
import numpy as np
from datasets import GameReplayDataset
from networks import Repr_Net, Dynamics_Net, Predict_Net
from datetime import datetime
from torch.utils.data import DataLoader
from torch import optim, nn

MAX_BUFFER_SIZE = 1000
BUFFER = os.listdir('game_data/')
BUFFER = np.sort(BUFFER)
DATASET_SIZE = 10000
REPR_WEIGHTS_PATH = "./repr_weights"
DYNAMIC_WEIGHTS_PATH = "./dynamic_weights"
PREDICT_WEIGHTS_PATH = "./predict_weights"
ID1 = "repr_net"
ID2 = "dynamics_net"
ID3 = "predict_net"
BATCH_SIZE = 1024
STATE_DIMS = [17,8,8]
POLICY_SIZE = 512
DEVICE = "cuda:0"
K = 5
REPR_LR = .00001*K
DYNAMICS_LR = .00001
PREDICT_LR = .00001
NUM_EPOCHS = 1000
LR_DECAY_STEPS = 100
LR_DECAY_RATE = .9


def update_buffer(BUFFER=None):
	BUFFER = np.sort(BUFFER) # Buffer size shouldn't get too big so we should be fine
	if len(BUFFER) > MAX_BUFFER_SIZE:
		num_to_remove = MAX_BUFFER_SIZE - len(BUFFER)
		for _ in range(num_to_remove):
			os.remove('game_data/' + BUFFER[0])
	return BUFFER

def get_latest_weights(weights_path):
	weights_list = os.listdir(weights_path)
	weights_size = len(weights_list)
	if False:
		weights_list = np.sort(weights_list)
		return weights_path + "/" + weights_list[0]
	else:
		return None

def get_latest_networks():
	# Initialize Repr_Net
	repr_net = Repr_Net().to(DEVICE)
	weights = get_latest_weights(REPR_WEIGHTS_PATH)
	if weights is not None:
		repr_net.load_state_dict(torch.load(weights))

	# Initialize Dynamics_Net
	dynamics_net = Dynamics_Net().to(DEVICE)
	weights = get_latest_weights(DYNAMIC_WEIGHTS_PATH)
	if weights is not None:
		dynamics_net.load_state_dict(torch.load(weights))

	# Initialize Predict_Net
	predict_net = Predict_Net().to(DEVICE)
	weights = get_latest_weights(PREDICT_WEIGHTS_PATH)
	if weights is not None:
		predict_net.load_state_dict(torch.load(weights))

	return repr_net, dynamics_net, predict_net

# Global function
def get_a_rep(action):
	"""
	Interprets the action (an int) and changes that to a selected piece and selected move
	:param action:
	:return:
	"""
	# action_tensor = torch.zeros((2, 8, 8))
	# unraveled_piece_pos = action // 8
	# selected_piece = (unraveled_piece_pos // 8, unraveled_piece_pos % 8)
	#
	# unraveled_move_pos = action % 8
	# i, j = self.pos_map[unraveled_move_pos]
	#
	# selected_move = (selected_piece[0] + i, selected_piece[1] + j)
	# action_tensor[0, i, j] = selected_piece
	# action_tensor[1, i, j] = selected_move
	if type(action) is int:
		size = 1
	elif type(action) is float:
		raise TypeError("Action type input should be int or tensor")
	else:
		size = action.shape[0]
		action = action.to(torch.long).detach()

	action_tensor = torch.zeros((size, 512))
	action_tensor[torch.arange(size), action] = 1.
	action_tensor = action_tensor.reshape(size, 8, 8, 8)

	return action_tensor

def main():
	# Define datasets and networks
	replay_dataset = GameReplayDataset(size=DATASET_SIZE)
	data_loader = DataLoader(replay_dataset, batch_size=BATCH_SIZE, shuffle=True)
	repr_net, dynamics_net, predict_net = get_latest_networks()

	# Define optimizers
	repr_optimizer = optim.Adam(repr_net.parameters(), lr=REPR_LR, weight_decay=0.0001)
	dynamics_optimizer = optim.Adam(dynamics_net.parameters(), lr=DYNAMICS_LR, weight_decay=0.0001)
	predict_optimizer = optim.Adam(predict_net.parameters(), lr=PREDICT_LR, weight_decay=0.0001)

	# https://discuss.pytorch.org/t/how-to-optimize-multi-models-parameter-in-one-optimizer/3603
	# optimizer = optim.Adam(list(repr_net.parameters()) + list(dynamics_net.parameters()) + list(predict_net.parameters()), lr=.00001, weight_decay=.0001)
	logsoftmax = nn.LogSoftmax()

	for epoch in range(NUM_EPOCHS):
		losses = []

		# Learning rate decay
		# if epoch % 100 == 0 and epoch != 0:
		# 	REPR_LR /= 10
		# 	DYNAMICS_LR /= 10
		# 	PREDICT_LR /= 10
		# 	print(REPR_LR, DYNAMICS_LR, PREDICT_LR)

		REPR_LR *= LR_DECAY_RATE**(epoch/LR_DECAY_STEPS)
		DYNAMICS_LR *= LR_DECAY_RATE**(epoch/LR_DECAY_STEPS)
		PREDICT_LR *= LR_DECAY_RATE**(epoch/LR_DECAY_STEPS)
		print("Learning rates:", REPR_LR, DYNAMICS_LR, PREDICT_LR)

		# Define optimizers
		repr_optimizer = optim.Adam(repr_net.parameters(), lr=REPR_LR, weight_decay=0.0001)
		dynamics_optimizer = optim.Adam(dynamics_net.parameters(), lr=DYNAMICS_LR, weight_decay=0.0001)
		predict_optimizer = optim.Adam(predict_net.parameters(), lr=PREDICT_LR, weight_decay=0.0001)

		# optimizer = optim.Adam(list(repr_net.parameters()) + list(dynamics_net.parameters()) + list(predict_net.parameters()), lr=.00001, weight_decay=.0001)
		for batch_idx, game_part in enumerate(data_loader):

			# Reshape batches to the correct sizes
			state = game_part[:, 0, 4 + POLICY_SIZE:].reshape([-1] + STATE_DIMS).to(DEVICE)
			actions = game_part[:, :K, 1].reshape(-1, K).to(DEVICE)
			rrp = game_part[:, :K + 1, 2:4 + POLICY_SIZE].to(DEVICE)

			# Set optimizers to zero grad
			repr_optimizer.zero_grad()
			dynamics_optimizer.zero_grad()
			predict_optimizer.zero_grad()
			# optimizer.zero_grad()
			loss = 0

			# Get the initial state representation
			s_rep = repr_net(state)
			# Scale the representation so it is in the same range as the action input
			state_shape = s_rep.shape
			flattened_s_rep = (s_rep.view(state_shape[0], -1) - torch.min(s_rep.view(state_shape[0], -1), dim=0)[0]) \
												/ (torch.max(s_rep.view(state_shape[0], -1), dim=0)[0] -
													 torch.min(s_rep.view(state_shape[0], -1), dim=0)[0])
			s_rep = flattened_s_rep.view(*state_shape)

			for i in range(K):
				mcts_policy = rrp[:,i,2:]
				true_value = rrp[:,i,0]
				action_dist, value = predict_net(s_rep)
				# Get the policy loss with cross-entropy
				policy_loss = torch.mean(torch.sum(- mcts_policy * logsoftmax(action_dist), 1))
				# policy_loss = torch.mean((mcts_policy - action_dist)**2) # Try MSE loss here. That might improve things
				# Get value loss with MSE
				value_loss = torch.mean((value - true_value)**2)

				loss += value_loss + policy_loss # Add a reward loss here later

				# Transfer to the next state with dynamics_net
				true_action = actions[:, i]
				a_rep = get_a_rep(true_action).to(DEVICE)

				new_s_rep, reward = dynamics_net(torch.cat([s_rep, a_rep], dim=1))

				# Scale the representation so it is in the same range as the action input
				state_shape = new_s_rep.shape
				flattened_s_rep = (new_s_rep.view(state_shape[0],-1) - torch.min(new_s_rep.view(state_shape[0],-1), dim=0)[0]) \
													/ (torch.max(new_s_rep.view(state_shape[0], -1), dim=0)[0]- torch.min(new_s_rep.view(state_shape[0], -1), dim=0)[0])
				s_rep = flattened_s_rep.view(*state_shape)

			loss /= K
			losses.append(loss.item())

			loss.backward()

			for p in dynamics_net.parameters():
				if p.grad is not None:
					p.grad *= .5  # Scale the dynamics net gradient by .5

			# optimizer.step()
			repr_optimizer.step()
			dynamics_optimizer.step()
			predict_optimizer.step()

			# print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTotal Loss: {:.6f}'.format(
			# 	epoch, batch_idx * BATCH_SIZE, len(data_loader.dataset), 100. * batch_idx * BATCH_SIZE / len(data_loader.dataset),
			# 	loss.item()))
		print('Train Epoch: {} \tAverage Loss: {:.6f}'.format(epoch, np.mean(losses)))

		# Save the current weights
		if epoch % 100 == 0 and epoch != 0:
			print("Saving weights...")
			torch.save(repr_net.state_dict(), REPR_WEIGHTS_PATH + "/repr_weights_" + "_".join(str(datetime.now()).split()))
			torch.save(dynamics_net.state_dict(), DYNAMIC_WEIGHTS_PATH + "/dynamic_weights_" + "_".join(str(datetime.now()).split()))
			torch.save(predict_net.state_dict(), PREDICT_WEIGHTS_PATH + "/predict_weights_" + "_".join(str(datetime.now()).split()))


if __name__=='__main__':
	main()


