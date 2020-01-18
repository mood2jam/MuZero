import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, channels=32):
    super().__init__()
    self.block = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                               nn.BatchNorm2d(channels),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(channels, channels, 3, padding=1),
                               nn.BatchNorm2d(channels))
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
    residual = x
    x = self.block(x)
    x += residual
    x = self.activation(x)
    return x

class ConvBlock(nn.Module):
  def __init__(self, in_channels=17, out_channels=32):
    super().__init__()
    self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                              nn.BatchNorm2d(out_channels),
                              nn.ReLU(inplace=True))

  def forward(self, x):
    return self.block(x)

class Repr_Net(nn.Module):
  def __init__(self, start_channels=6, res_block_channels=32, num_res_blocks=3):
    super(Repr_Net, self).__init__()
    self.res_blocks = nn.Sequential(*[ResidualBlock(res_block_channels) for _ in range(num_res_blocks)])
    self.conv_block = ConvBlock(start_channels,res_block_channels)
  def forward(self, x):
    out = self.conv_block(x)
    out = self.res_blocks(out)
    return out

class Dynamics_Net(nn.Module):
  def __init__(self, num_res_blocks=3, action_channels=8, res_block_channels=32, image_size=8, intermediate_rewards=False):
    super(Dynamics_Net, self).__init__()
    assert res_block_channels >= 16
    self.res_blocks = nn.Sequential(*[ResidualBlock(res_block_channels) for _ in range(num_res_blocks)])
    self.conv_block = nn.Sequential(nn.Conv2d(action_channels + res_block_channels, res_block_channels, 1),
                                    nn.BatchNorm2d(res_block_channels),
                                    nn.ReLU(inplace=True))
    self.reward_block = nn.Sequential(nn.Conv2d(res_block_channels, res_block_channels//16, 1),
                                      nn.BatchNorm2d(res_block_channels//16),
                                      nn.ReLU(inplace=True))
    self.fc1 = nn.Linear(res_block_channels//16*image_size*image_size, 8)
    self.fc2 = nn.Linear(8, 1)
    self.activation = nn.ReLU(inplace=True)
    self.intermediate_rewards = intermediate_rewards
    self.tanh = nn.Tanh()
    self.image_size = image_size
    self.res_block_channels = res_block_channels

  def forward(self, x):
    out = self.conv_block(x)
    state = self.res_blocks(out)

    if self.intermediate_rewards:
      out = self.reward_block(out)
      out = out.view(-1, (self.res_block_channels//16)*self.image_size*self.image_size)
      out = self.fc1(out)
      out = self.activation(out)
      out = self.fc2(out)
      reward = self.tah(out)
    else:
      reward = torch.zeros((state.shape[0], 1))

    return state, reward

class Predict_Net(nn.Module):
  def __init__(self, num_res_blocks=5, num_actions=256, res_block_channels=32, image_size=8):
    super(Predict_Net, self).__init__()
    assert res_block_channels >= 16
    # assert image_size*image_size*res_block_channels == num_actions
    self.res_blocks = nn.Sequential(*[ResidualBlock(res_block_channels) for _ in range(num_res_blocks)])
    self.action_head = nn.Sequential(nn.Conv2d(res_block_channels, res_block_channels//4, 1))
                                      # nn.BatchNorm2d(res_block_channels//4),
                                      # nn.ReLU(inplace=True))
    self.value_head = nn.Sequential(nn.Conv2d(res_block_channels, res_block_channels//16, 1),
                                      nn.BatchNorm2d(res_block_channels//16),
                                      nn.ReLU(inplace=True))
    self.fc1 = nn.Linear((res_block_channels//16) * image_size * image_size, 8)
    self.fc2 = nn.Linear(8, 1)
    self.activation = nn.ReLU(inplace=True)
    self.tanh = nn.Tanh()
    self.num_actions=num_actions
    self.softmax = nn.Softmax(dim=1)
    self.image_size = image_size
    self.res_block_channels = res_block_channels

  def forward(self, x):
    out = self.res_blocks(x)
    actions = self.action_head(out)
    actions = actions.view(-1, self.num_actions)

    value = self.value_head(out)
    value = value.view(-1, self.res_block_channels//16 * self.image_size * self.image_size)
    value = self.fc1(value)
    self.activation(value)
    value = self.fc2(value)

    return self.softmax(actions), self.tanh(value)



if __name__ == '__main__':
  # Check to make sure our networks are working
  rn = Repr_Net()
  input = torch.randn((1,17,8,8))
  print("Output from Repr_Net:", rn(input).shape)
  print(rn)
  print()

  dn = Dynamics_Net()
  input = torch.randn((1,40,8,8))
  state, reward = dn(input)
  print("Output from Dynamics_Net:", state.shape, reward.shape)
  print(dn)
  print()

  pn = Predict_Net()
  input = torch.randn((1,32,8,8))
  actions, values = pn(input)
  print("Output from Predict_Net:", actions.shape, values.shape)
  print(pn)

