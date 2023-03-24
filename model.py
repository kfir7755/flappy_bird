import torch
from torch.nn import MSELoss, Conv2d, MaxPool2d, Flatten, Module, Linear
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import numpy as np

BATCH_SIZE = 32


class DQN(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(3, 16, kernel_size=4)
        self.maxPool1 = MaxPool2d((4, 3))
        self.conv2 = Conv2d(16, 32, kernel_size=2)
        self.maxPool2 = MaxPool2d((3, 2))
        # flatten
        self.linear1 = Linear(512, 128)
        self.linear2 = Linear(128, 32)
        self.linear3 = Linear(32, 8)
        self.linear4 = Linear(8, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.maxPool2(x)
        x = torch.flatten(x)
        x = torch.reshape(x, (int(x.size(dim=0)/512), 512))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return F.softmax(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.int)
        # (n, x)

        if len(state.shape) == 3:
            # need to make (3,64,32) to (1,3,64,32)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)
        # if pred.dim() == 1:
        #     pred = torch.unsqueeze(pred, 0)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][int(action[idx])] = Q_new

        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

    def train_long_memory(self, D):
        if len(D) > BATCH_SIZE:
            mini_sample = random.sample(D, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = D

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, dones)
