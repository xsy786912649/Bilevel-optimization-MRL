from train import *

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from extra_function import *
import random
from meta_policy import *
import csv

if dis_i==2:
    lambda1 =0.2
elif dis_i==1:
    lambda1 =0.1

total_rewards_alltask_0=[]
total_rewards_alltask_1=[]
total_rewards_alltask_2=[]
total_rewards_alltask_3=[]
total_rewards_alltask_4=[]

for num_tasks in range(task_number):

    map_name = np.load('maps/map'+str(num_tasks)+'.npy')
    map_name = map_name.tolist()
    env = gym.make("FrozenLake-v1",desc= map_name, is_slippery=False)

    if dis_i==1:
        task_specific_theta = training_meta_theta_1
    else:
        task_specific_theta = training_meta_theta_2

    '''
    task_specific_theta=torch.zeros((16, 4))
    lambda1=lambda1*2
    '''

    total_rewards=[]
    print("--------------")
    for i in range(5):
        total_reward,observations,qtable_meta_policy = sample_trajectorie(env, gamma, task_specific_theta)
        print(i," times update: ","total_reward: ",total_reward)

        task_specific_policy=one_step_adaptation(task_specific_theta,qtable_meta_policy,lambda1,dis_i) 
        task_specific_theta=torch.log(task_specific_policy) 
        total_rewards.append(total_reward)
    total_rewards_alltask_0.append(total_rewards[0])
    total_rewards_alltask_1.append(total_rewards[1])
    total_rewards_alltask_2.append(total_rewards[2])
    total_rewards_alltask_3.append(total_rewards[3])
    total_rewards_alltask_4.append(total_rewards[4])

print(np.average(np.array(total_rewards_alltask_0)))
print(np.average(np.array(total_rewards_alltask_1)))
print(np.average(np.array(total_rewards_alltask_2)))
print(np.average(np.array(total_rewards_alltask_3)))
print(np.average(np.array(total_rewards_alltask_4)))


if dis_i==2:
    with open("./results/result_hole_d2.csv","a+") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([np.average(np.array(total_rewards_alltask_0)),np.average(np.array(total_rewards_alltask_1)),np.average(np.array(total_rewards_alltask_2)),np.average(np.array(total_rewards_alltask_3)),np.average(np.array(total_rewards_alltask_4))])
elif dis_i==1:
    with open("./results/result_hole_d1.csv","a+") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([np.average(np.array(total_rewards_alltask_0)),np.average(np.array(total_rewards_alltask_1)),np.average(np.array(total_rewards_alltask_2)),np.average(np.array(total_rewards_alltask_3)),np.average(np.array(total_rewards_alltask_4))])

'''
if dis_i==2:
    with open("./results/result_hole_d2_from0.csv","a+") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([np.average(np.array(total_rewards_alltask_0)),np.average(np.array(total_rewards_alltask_1)),np.average(np.array(total_rewards_alltask_2)),np.average(np.array(total_rewards_alltask_3)),np.average(np.array(total_rewards_alltask_4))])
elif dis_i==1:
    with open("./results/result_hole_d1_from0.csv","a+") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow([np.average(np.array(total_rewards_alltask_0)),np.average(np.array(total_rewards_alltask_1)),np.average(np.array(total_rewards_alltask_2)),np.average(np.array(total_rewards_alltask_3)),np.average(np.array(total_rewards_alltask_4))])
'''
