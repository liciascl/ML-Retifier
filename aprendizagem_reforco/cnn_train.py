from stable_baselines3.common.env_util import make_vec_env
from ambiente import Simulador

import gym
import torch as th
import torch.nn as nn
import pandas as pd
import numpy as np
from stable_baselines3 import A2C

# Parallel environments
env = make_vec_env('CartPole-v1', n_envs=4)



env = Simulador()
learning_data = {'state' : [], 'reward' : [], 'RL' : []}
test_data = {'action' : [], 'obs' : [], 'reward' : [], 'state' : []}

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save("a2c_cartpole")

del model # remove to demonstrate saving and loading

model = A2C.load("a2c_cartpole")

obs = env.reset()
while True:
	rl = np.random.uniform(8000, 10000)
	state, reward, done, info = env.step(rl)
	print(state, reward, done)
	learning_data['state'].append(state)
	learning_data['reward'].append(reward)
	learning_data['RL'].append(rl)

stable_baselines3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=20, deterministic=True, render=False, callback=None, reward_threshold=None, return_episode_rewards=False, warn=True)

df = pd.DataFrame.from_dict(learning_data, orient='index').T.to_csv('output_test_data.csv',index=True)
#df2 = pd.DataFrame.from_dict(learning_data, orient='index').T.to_csv('output_learning_data.csv',index=True)
