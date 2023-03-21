import time
import flappy_bird_gym
import numpy as np
import torch
from collections import deque
from model import *

myModel = Linear_QNet(2, 256, 2)
mytrainer = QTrainer(myModel, lr=0.001, gamma=0.95)
env = flappy_bird_gym.make("FlappyBird-v0")
old_obs = env.reset()
prev_frame_score = 0
epochs_num = 0
D = deque(maxlen=1000)
record = 0

while epochs_num < 1000:
    epsilon = 500 - epochs_num
    rand = np.random.randint(low=0, high=1250)
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        Q = myModel(torch.tensor(old_obs, dtype=torch.float))
        action = torch.argmax(Q).detach().numpy()
        action = int(action)
    # Processing:
    new_obs, reward, done, info = env.step(action)
    mytrainer.train_step(old_obs, action, reward, new_obs, done)
    this_frame_score = info['score']
    if this_frame_score > prev_frame_score:
        reward = 10
    elif done:
        reward = -10
    else:
        reward = 0
    D.append((old_obs, action, reward, new_obs, done))
    # Rendering the game:
    env.render()
    # FPS
    if epochs_num > 500:
        time.sleep(1 / 100)
    else:
        time.sleep(1 / 1000)
    old_obs = new_obs
    prev_frame_score = this_frame_score
    # Checking if the player is still alive
    if done:
        old_obs = env.reset()
        mytrainer.train_long_memory(D)
        epochs_num += 1
        prev_frame_score = 0
        if info['score'] > record:
            record = info['score']
        print(f'epoch number:{epochs_num},', info, ' record =', record)

env.close()
