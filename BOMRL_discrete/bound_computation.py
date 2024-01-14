from train import *

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from extra_function import *
import random

from meta_policy import *

task_specific_policies=[]
total_rewards=[]
kls=[]

if dis_i==2:
    lambda1 =0.5
elif dis_i==1:
    lambda1 =0.1

for num_tasks in range(task_number):

    map_name = np.load('maps/map'+str(num_tasks)+'.npy')
    map_name = map_name.tolist()
    env = gym.make("FrozenLake-v1",desc= map_name, is_slippery=False)

    if dis_i==1:
        task_specific_theta = training_meta_theta_1
    else:
        task_specific_theta = training_meta_theta_2

    print("----------------")
    for i in range(10):
        total_reward,observations,qtable_meta_policy = sample_trajectorie(env, gamma, task_specific_theta)
        print("total_reward: ",total_reward)
        if i==0:
            observations_meta=observations

        task_specific_policy=one_step_adaptation(task_specific_theta,qtable_meta_policy,lambda1,dis_i) 
        task_specific_theta=torch.log(task_specific_policy) 

    task_specific_policies.append(task_specific_policy)
    total_rewards.append(total_reward)

    if dis_i==1:
        meta_policy=torch.softmax(training_meta_theta_1,dim=1)
    else:
        meta_policy=torch.softmax(training_meta_theta_2,dim=1)

    if dis_i==2:     
        kl_this=KL(meta_policy,task_specific_policy,observations_meta)
        print(kl_this)
        kls.append(kl_this)
    if dis_i==1:
        kl_this=KL(task_specific_policy,meta_policy,observations_meta)
        print(kl_this)
        kls.append(kl_this)

print(np.average(np.array(total_rewards)))
print(np.average(np.array(kls)))