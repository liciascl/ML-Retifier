from stable_baselines3.common.env_util import make_vec_env
from env import RandomCircuits

import gym
import torch as th
import torch.nn as nn
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import numpy as np

env = RandomCircuits()
learning_data = {'state' : [], 'reward' : [], 'RL' : []}
test_data = {'action' : [], 'obs' : [], 'reward' : [], 'state' : []}
print(env.observation_space)
state = env.reset()
iterations = 1
for i in range(iterations):
	done = False
	while not done:
		rl = np.random.uniform(40000, 50000)
		state, reward, done, info = env.step(rl)
		print(state, reward, done)
		learning_data['state'] = np.append(state)
		learning_data['reward'] = np.append(reward)
		learning_data['RL'] = np.append(rl)
model = PPO("CnnPolicy", policy_kwargs=reward, verbose=1)
env.step(model.learn(1000))

print("Time to test")
obs = env.reset()

n_steps = 20
for step in range(n_steps):
	action, _ = model.predict(obs, deterministic=True)
	print("Step {}".format(step + 1))
	print("Action: ", action)
	obs, reward, done, info = env.step(action)
	print('obs=', obs, 'reward=', reward, 'done=', done)
	test_data['action'] = np.append(action)
	test_data['obs'] = np.append(obs)
	test_data['reward'] = np.append(reward)
	test_data['state'] = np.append(state)

stable_baselines3.common.evaluation.evaluate_policy(model, env, n_eval_episodes=20, deterministic=True, render=False, callback=None, reward_threshold=None, return_episode_rewards=False, warn=True)

df = pd.DataFrame.from_dict(test_data, orient='index').T.to_csv('output_test_data.csv',index=True)
df2 = pd.DataFrame.from_dict(learning_data, orient='index').T.to_csv('output_learning_data.csv',index=True)
