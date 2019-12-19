from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle


class GameReplayDataset(Dataset):
    """Game Replay dataset."""

    def __init__(self, root_dir='./game_data', size = 10000, transform=None, state_dims = (17,8,8), K=5, policy_size=512):
        """
        Args:
            root_dir (string): Directory with all the games.
            size (int): Number of games to sample from.
        """
        self.size = size
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)
        self.indices = np.random.choice(len(self.files), self.size, replace=True)
        self.K = 5
        self.state_dims = state_dims
        self.policy_size = policy_size

        # Code from https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
    def load_game(self, game_file):
        with open(self.root_dir + '/' + game_file, 'rb') as handle:
          shape, indices, values = pickle.load(handle)
        loaded_game = torch.zeros(shape)
        loaded_game[indices[:, 0], indices[:, 1]] = values
        return loaded_game

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Get the game at index
        # print(idx)
        # print(self.indices)
        game = self.load_game(self.files[self.indices[idx]])

        L = game.shape[0]
        # Select a random state from the game
        i = np.random.choice(L)

        # Gets state, actions, return/reward/policy tuple
        # state = game[i, 4 + self.policy_size:]
        # actions = game[i:i + self.K, 1]
        # rrp = game[i:i + self.K + 1, 2:4 + self.policy_size]

        game_part = game[i:i+self.K+1,:]

        # Handles the case were we got to the end of the game
        while game_part.shape[0] < self.K + 1:
            # Make an absorbing state
            game_part = torch.cat([game_part, game_part[-1,:].unsqueeze(0)], dim=0)
        assert game_part.shape[0] == self.K + 1

        return game_part


