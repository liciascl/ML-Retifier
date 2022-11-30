import yaml
import os
import datetime
import numpy as np
import time
from model import A2C

def log(message):
    print('[Simulação {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) + "] " + message)


log('Starting to train the agent ..')

all_rewards = []
learner = A2C(1)
training_start_time = time.time()
for i in range(options['episodes']):
    log('Episode: ' + str(i+1))
    start = time.time()
    total_reward = learner.train_episode()
    end = time.time()
    all_rewards.append(total_reward)
    log('Episode: ' + str(i) + ' - done with total reward = ' + str(total_reward))
    log('Episode ' + str(i) + ' Run Time ~ ' + str((start - end) / 60) + ' minutes.')
    print('')
training_end_time = time.time()
log('Total Training Run Time ~ ' + str((training_end_time - training_start_time) / 60) + ' minutes.')

mean_reward = np.mean(all_rewards[-100:])
