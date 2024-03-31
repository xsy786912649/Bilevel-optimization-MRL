import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from extra_function_nohole import *
import random
import sys

dis_i=int(sys.argv[1])
if dis_i==2:
    lambda1 =4.0
elif dis_i==1:
    lambda1 =4.0

gamma = 0.8
task_number=20

def KL_1(a, b):
    a=torch.clip(a,0.0001,1.0)
    b=torch.clip(b,0.0001,1.0)
    cc=torch.sum(a * torch.log(a) - a * torch.log(b))
    return cc

def one_step_adaptation(meta_phi,qtable_meta_policy,lambda1,d_i=1):
    if d_i==2:
        task_specific_theta=meta_phi.data+1/lambda1*qtable_meta_policy.data
        task_specific_policy=torch.softmax(task_specific_theta,dim=1)
    elif d_i==1:
        meta_policy=torch.softmax(meta_phi.data,dim=1)
        A_table_meta_policy=qtable_meta_policy-torch.sum(meta_policy*qtable_meta_policy,dim=1).reshape((16,1))
        task_specific_policy_list=[]
        for i in range(16):
            ccc=meta_phi[i].detach().clone().requires_grad_(True)
            optimizer111 = torch.optim.SGD([ccc], lr=0.3*2)
            optimizer111.zero_grad() 
            for j in range(20):
                optimizer111.zero_grad()
                loss=torch.sum(-torch.softmax(ccc,dim=0)*A_table_meta_policy[i]/lambda1+ KL_1( meta_policy[i].detach().clone(),torch.softmax(ccc,dim=0)) )
                loss.backward()
                optimizer111.step()
                optimizer111.zero_grad()
            task_specific_policy_list.append(torch.softmax(ccc,dim=0).detach().data.reshape((-1)))

        task_specific_policy=torch.stack(task_specific_policy_list, dim = 0).detach().clone()
    return task_specific_policy

def meta_gradient(task_specific_policy,task_specific_observations,A_table_task_specific,d_i=1):
    if d_i==2:
        return (task_specific_observations * task_specific_policy)*A_table_task_specific
    elif d_i==1:
        return (task_specific_observations * task_specific_policy)*A_table_task_specific
    else:
        return None

if __name__ == "__main__":

    meta_phi = torch.zeros((16, 4),requires_grad=True)
    meta_phi.data=meta_phi.data+0.1
    #optimizer = torch.optim.SGD([meta_phi], lr=1.0*5)
    optimizer = torch.optim.Adam([meta_phi], lr=0.03)

    for i_episode in range(150):
        print("i_episode: ",i_episode)
        optimizer.zero_grad() 
        grads_update=None

        task_list=random.sample(list(range(task_number)),5)
        print(task_list)
        for num_tasks in task_list:

            reward_map = np.load('maps_nohole/map'+str(num_tasks)+'.npy')
            env = gym.make("FrozenLake-v1",desc= ['SFFF', 'FFFF', 'FFFF', 'FFFG'], is_slippery=False)

            total_reward,observations,qtable_meta_policy = sample_trajectorie(reward_map,env, gamma, meta_phi)
            print("total_reward (before adatation): ",total_reward)
            
            task_specific_policy=one_step_adaptation(meta_phi,qtable_meta_policy,lambda1,dis_i) 
            task_specific_theta=torch.log(task_specific_policy)+2.0
            task_specific_total_reward,task_specific_observations,task_specific_qtable = sample_trajectorie(reward_map,env, gamma, task_specific_theta)
            A_table_task_specific=task_specific_qtable-torch.sum(task_specific_policy*task_specific_qtable,dim=1).reshape((16,1))
            print("total_reward (after adatation): ",task_specific_total_reward)
            
            grad1= meta_gradient(task_specific_policy,task_specific_observations,A_table_task_specific,dis_i)
            if grads_update==None:
                grads_update=grad1.data/5.0
            else:
                grads_update+=grad1.data/5.0
            
        optimizer.zero_grad() 
        meta_phi.grad= -grads_update
        optimizer.step()
        optimizer.zero_grad()

        print(meta_phi)


