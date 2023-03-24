import time
import gym_ple
import numpy as np
from collections import deque
from model import *
from gym.wrappers import TransformObservation
import cv2


def resize_state(state):
    ret = cv2.resize(state, dsize=(32, 64), interpolation=cv2.INTER_AREA)
    ret = np.swapaxes(ret, 2, 1)
    ret = np.swapaxes(ret, 1, 0)
    return ret/255


myModel = DQN()
mytrainer = QTrainer(myModel, lr=0.01, gamma=0.99)
env = gym_ple.make("FlappyBird-v0")
# env = TransformObservation(env, lambda x: x.swapaxes(-1, 0))
old_obs = env.reset()
old_obs = resize_state(old_obs)
epochs_num = 0
D = deque(maxlen=1000)
record = 0
score = 0
while epochs_num < 200:
    epsilon = 80 - epochs_num
    rand = np.random.randint(low=0, high=200)
    if rand < epsilon:
        action = env.action_space.sample()
    else:
        Q = myModel(torch.tensor(old_obs, dtype=torch.float))
        action = torch.argmax(Q).detach().numpy()
        action = int(action)
    # Processing:
    new_obs, reward, done, info = env.step(action)
    new_obs = resize_state(new_obs)
    mytrainer.train_step(old_obs, action, reward, new_obs, done)
    if reward > 0:
        score += 1
    if reward < 0:
        reward = -15
    D.append((old_obs, action, reward, new_obs, done))
    # Rendering the game:
    env.render()
    # FPS
    # if epochs_num > 500:
    time.sleep(1 / 2000)
    # else:
    #     time.sleep(1 / 30)
    old_obs = new_obs
    # Checking if the player is still alive
    if done:
        old_obs = env.reset()
        old_obs = resize_state(old_obs)
        mytrainer.train_long_memory(D)
        epochs_num += 1
        if score > record:
            record = score
        print(f'epoch number:{epochs_num}, score =', score, ' record =', record)
        score = 0
env.close()
