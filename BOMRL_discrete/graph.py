import numpy as np
import time
import torch
from torch import nn 
from torch.utils.data import Dataset,DataLoader,TensorDataset
from torch.nn import functional as F

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, random, time
import csv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
default_cycler = (cycler(color=['#295778', '#62c5cc', '#ee7663', '#f3b554', '#577829']) )  
plt.rc('axes', prop_cycle=default_cycler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

iteration_number=(np.array(list(range(5))))


data_no_2 = np.loadtxt("./results/result_nohole_d2.csv", delimiter=',')
data_no_2_mean=np.sum(data_no_2,axis=0)/data_no_2.shape[0]
data_no_2_sd=np.sqrt(np.var(data_no_2,axis=0))*4.0

data_no_1 = np.loadtxt("./results/result_nohole_d1.csv", delimiter=',')
data_no_1_mean=np.sum(data_no_1,axis=0)/data_no_1.shape[0]
data_no_1_sd=np.sqrt(np.var(data_no_1,axis=0))*4.0

data_hole_2 = np.loadtxt("./results/result_hole_d2.csv", delimiter=',')
data_hole_2_mean=np.sum(data_hole_2,axis=0)/data_hole_2.shape[0]
data_hole_2_sd=np.sqrt(np.var(data_hole_2,axis=0))*4.0

data_hole_1 = np.loadtxt("./results/result_hole_d1.csv", delimiter=',')
data_hole_1_mean=np.sum(data_hole_1,axis=0)/data_hole_1.shape[0]
data_hole_1_sd=np.sqrt(np.var(data_hole_1,axis=0))*4.0

data_no_2_from0 = np.loadtxt("./results/result_nohole_d2_from0.csv", delimiter=',')
data_no_2_from0_mean=np.sum(data_no_2_from0,axis=0)/data_no_2_from0.shape[0]
data_no_2_from0_sd=np.sqrt(np.var(data_no_2_from0,axis=0))*4.0

data_no_1_from0 = np.loadtxt("./results/result_nohole_d1_from0.csv", delimiter=',')
data_no_1_from0_mean=np.sum(data_no_1_from0,axis=0)/data_no_1_from0.shape[0]
data_no_1_from0_sd=np.sqrt(np.var(data_no_1_from0,axis=0))*4.0

data_hole_2_from0 = np.loadtxt("./results/result_hole_d2_from0.csv", delimiter=',')
data_hole_2_from0_mean=np.sum(data_hole_2_from0,axis=0)/data_hole_2_from0.shape[0]
data_hole_2_from0_sd=np.sqrt(np.var(data_hole_2_from0,axis=0))*4.0

data_hole_1_from0 = np.loadtxt("./results/result_hole_d1_from0.csv", delimiter=',')
data_hole_1_from0_mean=np.sum(data_hole_1_from0,axis=0)/data_hole_1_from0.shape[0]
data_hole_1_from0_sd=np.sqrt(np.var(data_hole_1_from0,axis=0))*4.0


data_hole_gd = np.loadtxt("./results/result_hole_gd.csv", delimiter=',')
data_hole_gd_mean=np.sum(data_hole_gd,axis=0)/data_hole_gd.shape[0]#-np.array([0.05,0.05,0.01,0.01,0.01])
data_hole_gd_sd=np.sqrt(np.var(data_hole_gd,axis=0))*4.0

data_nohole_gd = np.loadtxt("./results/result_nohole_gd.csv", delimiter=',')
data_nohole_gd_mean=np.sum(data_nohole_gd,axis=0)/data_nohole_gd.shape[0]#-0.01
data_nohole_gd_sd=np.sqrt(np.var(data_nohole_gd,axis=0))*4.0

data_no_2_optimal = np.loadtxt("./results/result_nohole_d2_optimal.csv", delimiter=',')
data_no_1_optimal= np.loadtxt("./results/result_nohole_d1_optimal.csv", delimiter=',')
data_hole_2_optimal = np.loadtxt("./results/result_hole_d2_optimal.csv", delimiter=',')
data_hole_1_optimal = np.loadtxt("./results/result_hole_d1_optimal.csv", delimiter=',')


plt.rcParams.update({'font.size': 24})
plt.rcParams['font.sans-serif']=['Arial']
plt.rcParams['axes.unicode_minus']=False 

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_hole_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL (ours)")
ax.fill_between(axis,data_hole_1_mean-data_hole_1_sd,data_hole_1_mean+data_hole_1_sd,alpha=0.4)
plt.plot(axis,data_hole_1_from0_mean,'-.',marker="x", linewidth=2.5,markersize=8,label="Random initialization")
ax.fill_between(axis,data_hole_1_from0_mean-data_hole_1_from0_sd,data_hole_1_from0_mean+data_hole_1_from0_sd,alpha=0.4)
plt.plot(axis,data_hole_gd_mean,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
ax.fill_between(axis,data_hole_gd_mean-data_hole_gd_sd,data_hole_gd_mean+data_hole_gd_sd,alpha=0.4)
plt.plot(axis,data_hole_1_optimal,'--', linewidth=2.5 ,label="Optimal task-specific policies")


#plt.xticks(np.arange(0,iterations,40))
plt.title('High task variance ($\mathcal{A l g}^{(1)}$ applied)',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Expected Accumulated reward",size=28)
plt.ylim(-0.5,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.129, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/hole_1.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_no_1_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL (ours)")
ax.fill_between(axis,data_no_1_mean-data_no_1_sd,data_no_1_mean+data_no_1_sd,alpha=0.4)
plt.plot(axis,data_no_1_from0_mean,'-.',marker="x", linewidth=2.5,markersize=8,label="Random initialization")
ax.fill_between(axis,data_no_1_from0_mean-data_no_1_from0_sd,data_no_1_from0_mean+data_no_1_from0_sd,alpha=0.4)
plt.plot(axis,data_nohole_gd_mean,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
ax.fill_between(axis,data_nohole_gd_mean-data_nohole_gd_sd,data_nohole_gd_mean+data_nohole_gd_sd,alpha=0.4)
plt.plot(axis,data_no_1_optimal,'--', linewidth=2.5 ,label="Optimal task-specific policies")
#plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Low task variance ($\mathcal{A l g}^{(1)}$ applied)',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Expected Accumulated reward",size=28)
plt.ylim(-0.25,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.115, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold')
plt.savefig('./figures/nohole_1.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_hole_2_mean,'-',marker="o",markersize=8, linewidth=2.5,label="BO-MRL (ours)")
ax.fill_between(axis,data_hole_2_mean-data_hole_2_sd,data_hole_2_mean+data_hole_2_sd,alpha=0.4)
plt.plot(axis,data_hole_2_from0_mean,'-.',marker="x",markersize=8, linewidth=2.5,label="Random initialization")
ax.fill_between(axis,data_hole_2_from0_mean-data_hole_2_from0_sd,data_hole_2_from0_mean+data_hole_2_from0_sd,alpha=0.4)
plt.plot(axis,data_hole_gd_mean,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
ax.fill_between(axis,data_hole_gd_mean-data_hole_gd_sd,data_hole_gd_mean+data_hole_gd_sd,alpha=0.4)
plt.plot(axis,data_hole_2_optimal,'--' , linewidth=2.5 ,label="Optimal task-specific policies")
#plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('High task variance ($\mathcal{A l g}^{(2)}$ applied)',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Expected Accumulated reward",size=28)
plt.ylim(-0.3,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.113, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/hole_2.pdf') 
plt.show()

axis=iteration_number
plt.figure(figsize=(8*1.1,6*1.1))
ax = plt.gca()
plt.plot(axis,data_no_2_mean,'-',marker="o",markersize=8, linewidth=2.5 ,label="BO-MRL (ours)")
ax.fill_between(axis,data_no_2_mean-data_no_2_sd,data_no_2_mean+data_no_2_sd,alpha=0.4)
plt.plot(axis,data_no_2_from0_mean,'-.',marker="x", linewidth=2.5,markersize=8,label="Random initialization")
ax.fill_between(axis,data_no_2_from0_mean-data_no_2_from0_sd,data_no_2_from0_mean+data_no_2_from0_sd,alpha=0.4)
plt.plot(axis,data_nohole_gd_mean,'-',marker="1",markersize=8, linewidth=2.5 ,label="MAML")
ax.fill_between(axis,data_nohole_gd_mean-data_nohole_gd_sd,data_nohole_gd_mean+data_nohole_gd_sd,alpha=0.4)
plt.plot(axis,data_no_2_optimal,'--', linewidth=2.5 ,label="Optimal task-specific policies")
#plt.plot(axis,boil_clean,linestyle=(0,(3, 1, 1, 1, 1, 1)),label="BOIL with MOML")

#plt.xticks(np.arange(0,iterations,40))
plt.title('Low task variance ($\mathcal{A l g}^{(2)}$ applied)',size=28)
plt.xlabel('Number of policy adaptation steps',size=28)
plt.ylabel("Expected Accumulated reward",size=28)
plt.ylim(-0.25,1.75)
#plt.legend(loc=4)
plt.legend(loc=0, numpoints=1)
plt.subplots_adjust(left=0.115, right=0.993, top=0.923, bottom=0.126)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
#plt.setp(ltext, fontsize=18,fontweight='bold') 
plt.savefig('./figures/nohole_2.pdf') 
plt.show()


bar_width = 0.5
line_width = 1



fig, ax = plt.subplots(figsize=(8*1.1,6*1.1))

y_optimal=1.6381797250192158
y_data = y_optimal - np.array([ data_hole_1_mean[0], data_hole_1_mean[1], 1.30])
#y_data = [ data_hole_1_mean[0], data_hole_1_mean[1], 1.30, 1.6381797250192158]
x_data = ('No\n adaptation', 'One-step\n of $\mathcal{A l g}^{(1)}$', 'One-step of\n policy gradient')
std_err=[data_hole_1_sd[0]/2,data_hole_1_sd[1]/2,data_hole_1_sd[1]/2] 

error_params=dict(elinewidth=4,capsize=5)

bar = plt.bar(x_data, y_data, width=bar_width, linewidth=line_width ,yerr=std_err,error_kw=error_params, color=['#577829', '#295778', '#ee7663' ], edgecolor='black')

plt.text(1-0.35, 0.2674698 * 9 * 0.1 + 0.05, 'Upper bound for\n one-step $\mathcal{A l g}^{(1)}$', fontsize=20)
plt.axhline(y=0.2474698 * 9 * 0.1 , color='black', linestyle='--', linewidth=4, zorder=3)

plt.title('High task variance ($\mathcal{A l g}^{(1)}$ applied)',size=28)
ax.set_ylabel("Expected Optimality Gap",size=28)
plt.subplots_adjust(left=0.115, right=0.983, top=0.923, bottom=0.131)
plt.savefig('./figures/hole_1_bound.pdf') 
plt.show()


fig, ax = plt.subplots(figsize=(8*1.1,6*1.1))

y_optimal=1.6381797250192158
y_data = y_optimal - np.array([ data_hole_2_mean[0], data_hole_2_mean[1], 1.30])
#y_data = [ data_hole_2_mean[0], data_hole_2_mean[1], 1.30, 1.6381797250192158]
x_data = ('No\n adaptation', 'One-step\n of $\mathcal{A l g}^{(2)}$', 'One-step of\n policy gradient')
std_err=[data_hole_2_sd[0]/2,data_hole_2_sd[1]/2,data_hole_2_sd[1]/2] 

error_params=dict(elinewidth=4,capsize=5)

bar = plt.bar(x_data, y_data, width=bar_width, linewidth=line_width ,yerr=std_err,error_kw=error_params, color=['#577829', '#295778', '#ee7663' ], edgecolor='black')

plt.text(1-0.35, 0.6193449 * 9 * 0.1 - 0.2, 'Upper bound for\n one-step $\mathcal{A l g}^{(2)}$', fontsize=20)
plt.axhline(y=0.6193449 * 9 * 0.1 , color='black', linestyle='--', linewidth=4, zorder=3)

plt.title('High task variance ($\mathcal{A l g}^{(2)}$ applied)',size=28)
ax.set_ylabel("Expected Optimality Gap",size=28)
plt.subplots_adjust(left=0.136, right=0.983, top=0.923, bottom=0.131)
plt.savefig('./figures/hole_2_bound.pdf') 
plt.show()



fig, ax = plt.subplots(figsize=(8*1.1,6*1.1))
y_optimal=1.6238083970192134

y_data = y_optimal - np.array([ data_no_1_mean[0], data_no_1_mean[1], 1.5031502591632093 ])
#y_data = [ data_no_1_mean[0], data_no_1_mean[1], 1.5031502591632093, 1.6238083970192134]
x_data = ('No\n adaptation', 'One-step\n of $\mathcal{A l g}^{(1)}$', 'One-step of\n policy gradient')
std_err=[data_no_1_sd[0]/2,data_no_1_sd[1]/2,data_no_1_sd[1]/2] 

error_params=dict(elinewidth=4,capsize=5)

bar = plt.bar(x_data, y_data, width=bar_width, linewidth=line_width ,yerr=std_err,error_kw=error_params, color=['#577829', '#295778', '#ee7663' ], edgecolor='black')

plt.text(1-0.35, 0.15265574 * 9* 0.1 +0.01  , 'Upper bound for\n one-step $\mathcal{A l g}^{(1)}$', fontsize=20)
plt.axhline(y=0.15265574 * 9* 0.1 , color='black', linestyle='--', linewidth=4, zorder=3)
plt.ylim(0.0,0.199)

plt.title('Low task variance ($\mathcal{A l g}^{(1)}$ applied)',size=28)
ax.set_ylabel("Expected Optimality Gap",size=28)
plt.subplots_adjust(left=0.136, right=0.983, top=0.923, bottom=0.131)
plt.savefig('./figures/nohole_1_bound.pdf') 
plt.show()


fig, ax = plt.subplots(figsize=(8*1.1,6*1.1))

y_optimal=1.6142083970192134
y_data = y_optimal - np.array([ data_no_2_mean[0], data_no_2_mean[1], 1.5031502591632093 ])
#y_data = [ data_no_2_mean[0], data_no_2_mean[1], 1.5031502591632093, 1.6142083970192134]
x_data = ('No\n adaptation', 'One-step\n of $\mathcal{A l g}^{(2)}$', 'One-step of\n policy gradient')
std_err=[data_no_2_sd[0]/2,data_no_2_sd[1]/2,data_no_2_sd[1]/2] 

error_params=dict(elinewidth=4,capsize=5)

bar = plt.bar(x_data, y_data, width=bar_width, linewidth=line_width ,yerr=std_err,error_kw=error_params, color=['#577829', '#295778', '#ee7663' ], edgecolor='black')


plt.text(1-0.35,  0.35228756 * 9.0* 0.05 - 0.025  , 'Upper bound for\n one-step $\mathcal{A l g}^{(2)}$', fontsize=20)
plt.axhline( y= 0.35228756 * 9.0* 0.05, color='black', linestyle='--', linewidth=4, zorder=3)
plt.ylim(0.0,0.18)

plt.title('Low task variance ($\mathcal{A l g}^{(2)}$ applied)',size=28)
ax.set_ylabel("Expected Optimality Gap",size=28)
plt.subplots_adjust(left=0.136, right=0.983, top=0.923, bottom=0.131)
plt.savefig('./figures/nohole_2_bound.pdf') 
plt.show() 


'''--------------------------------------------------'''
fig, ax = plt.subplots(figsize=(8*1.1,6*1.1))

y_optimal=1.6335099673006876
y_data = y_optimal - np.array([ 0.41384084713453, 1.31245, 1.3049245332733332 ])
#y_data = [0.41384084713453, 1.31245, 1.3049245332733332, 1.6335099673006876]
x_data = ('No\n adaptation', 'One-step\n of $\mathcal{A l g}^{(3)}$', 'One-step of\n policy gradient')
std_err=[data_hole_2_sd[0]/2,data_hole_2_sd[1]/2,data_hole_2_sd[1]/2] 

error_params=dict(elinewidth=4,capsize=5)

bar = plt.bar(x_data, y_data, width=bar_width, linewidth=line_width ,yerr=std_err,error_kw=error_params, color=['#577829', '#295778', '#ee7663' ], edgecolor='black')

plt.text(1-0.35, y_optimal-0.1933 - 0.28, 'Upper bound for\n one-step $\mathcal{A l g}^{(3)}$', fontsize=20)
plt.axhline(y=y_optimal-0.1933 , color='black', linestyle='--', linewidth=4, zorder=3)
plt.ylim(0.0,1.8)

plt.title('High task variance ($\mathcal{A l g}^{(3)}$ applied)',size=28)
ax.set_ylabel("Expected Optimality Gap",size=28)
plt.subplots_adjust(left=0.136, right=0.983, top=0.923, bottom=0.131)
plt.savefig('./figures/hole_3_bound.pdf') 
plt.show()


fig, ax = plt.subplots(figsize=(8*1.1,6*1.1))

y_optimal=1.6142083970192134
y_data = y_optimal - np.array([ 1.491573045339209, 1.51521, 1.5031502591632093 ])
#y_data = [ 1.491573045339209, 1.51521, 1.5031502591632093, 1.6142083970192134]
x_data = ('No\n adaptation', 'One-step\n of $\mathcal{A l g}^{(3)}$', 'One-step of\n policy gradient')
std_err=[data_no_2_sd[0]/2,data_no_2_sd[1]/2,data_no_2_sd[1]/2] 

error_params=dict(elinewidth=4,capsize=5)

bar = plt.bar(x_data, y_data, width=bar_width, linewidth=line_width ,yerr=std_err,error_kw=error_params, color=['#577829', '#295778', '#ee7663' ], edgecolor='black')


plt.text(1-0.35,  y_optimal-0.9132  - 0.12  , 'Upper bound for\n one-step $\mathcal{A l g}^{(3)}$', fontsize=20)
plt.axhline( y= y_optimal-0.9132 , color='black', linestyle='--', linewidth=4, zorder=3)
plt.ylim(0.0,0.799)

plt.title('Low task variance ($\mathcal{A l g}^{(3)}$ applied)',size=28)
ax.set_ylabel("Expected Optimality Gap",size=28)
plt.subplots_adjust(left=0.115, right=0.983, top=0.923, bottom=0.131)
plt.savefig('./figures/nohole_3_bound.pdf') 
plt.show() 

