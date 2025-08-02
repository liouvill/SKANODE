import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc as rc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
import scipy.io as sio
from skimage.metrics import structural_similarity as ssim


parser = argparse.ArgumentParser()
parser.add_argument('--npoints', type=int, default=600)
parser.add_argument('--freq', type=float, default=10)
args = parser.parse_args()

data = sio.loadmat("duffing_data.mat")

true_state = data['true_latent']
true_state = true_state[0,:args.npoints]

acc1_data_ = data['force']
acc2_data_ = data['obs']

rescaling = 1
acc1_data = rescaling*acc1_data_
acc2_data_ = rescaling*acc2_data_
acc2_data = acc2_data_[:args.npoints]

samp_ts_array = np.linspace(0., 1/args.freq*args.npoints , num = args.npoints, endpoint=False)





fig = plt.figure(figsize=[30, 15])
fig.subplots_adjust(hspace=0., wspace=0)

####################################################
sns.set_style('darkgrid')
rc('font', family='serif')
rc('text', usetex=True)
ax1 = plt.subplot(2,2,1)


names = ['skanode', 'sonode', 'anode']
labels =['SKANODE', 'SONODE', 'ANODE']
colors = ['#D15C6B', '#206491', '#B39B83']

def add_bit(x):
    iters = np.load('saved_models/'+names[x]+'/1/itr_arr.npy')
    loss_1 = np.load('saved_models/'+names[x]+'/1/loss_arr.npy')
    loss_2 = np.load('saved_models/'+names[x]+'/2/loss_arr.npy')
    loss_3 = np.load('saved_models/'+names[x]+'/3/loss_arr.npy')
    
    loss = np.empty((len(loss_1),3))
    for i in range(len(loss_1)):
        loss[i][0] = loss_1[i]
        loss[i][1] = loss_2[i]
        loss[i][2] = loss_3[i]
    
    loss_mean = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_mean[i] = np.mean(loss[i])
    
    loss_std = np.empty(len(loss_1))
    for i in range(len(loss_1)):
        loss_std[i] = np.std(loss[i])
        
    loss_p = loss_mean + loss_std
    loss_m = loss_mean - loss_std
    
    plt.plot(iters, loss_mean, color=colors[x], label=labels[x], lw=4)
    ax1.fill_between(x=iters, y1=loss_p, y2=loss_m, alpha=0.2, color=colors[x])



add_bit(2)
add_bit(1)
add_bit(0)

rc('font', family='serif')
rc('text', usetex=True)
plt.legend(fontsize=28, loc="upper right")
plt.xlabel('Epoch', fontsize=40)
plt.ylabel('MSE', fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(0, 5)
plt.title('Duffing Oscillator Training MSE', fontsize=40)


#################################################
sns.set_style('dark')
ax2 = plt.subplot(2,2,2)
skanode_to_plot_acc2 = np.load('saved_models/skanode/1/acc2_test.npy')
sonode_to_plot_acc2 = np.load('saved_models/sonode/1/acc2_test.npy')
anode_to_plot_acc2 = np.load('saved_models/anode/1/acc2_test.npy')
rc('font', family='serif')
rc('text', usetex=True)
plt.plot(samp_ts_array, sonode_to_plot_acc2, label='SONODE', color='#206491', linestyle='--', lw=3)
plt.plot(samp_ts_array, anode_to_plot_acc2, label='ANODE', color='#B39B83', linestyle='--', lw=3)
plt.plot(samp_ts_array, acc2_data[...,0], label='Ground-truth', color='black', lw=6)
plt.plot(samp_ts_array, skanode_to_plot_acc2, label='SKANODE', color='#D15C6B', linestyle='--', lw=4)
plt.plot(samp_ts_array, skanode_to_plot_acc2-acc2_data[...,0], label='difference', color='black', linestyle='-', lw=4)
plt.xlabel('Time', fontsize=40)
plt.ylabel('Acceleration', fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(-2.5, 3)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],\
           loc='upper right', fontsize=28, framealpha=0.8, ncol=2)

###################################################
sns.set_style('dark')
ax3 = plt.subplot(2,2,3)
skanode_to_plot_state = np.load('saved_models/skanode/1/state_test.npy')
sonode_to_plot_state = np.load('saved_models/sonode/1/state_test.npy')
anode_to_plot_state = np.load('saved_models/anode/1/state_test.npy')
rc('font', family='serif')
rc('text', usetex=True)
plt.plot(samp_ts_array, sonode_to_plot_state[...,0], label='SONODE', color='#206491', linestyle='--', lw=3)
plt.plot(samp_ts_array, anode_to_plot_state[...,0], label='ANODE', color='#B39B83', linestyle='--', lw=3)
plt.plot(samp_ts_array, true_state[...,0], label='Ground-truth', color='black', lw=6)
plt.plot(samp_ts_array, skanode_to_plot_state[...,0], label='SKANODE', color='#D15C6B', linestyle='--', lw=4)
plt.xlabel('Time', fontsize=40)
plt.ylabel('Displacement', fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(-0.5, 0.6)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],\
           loc='upper right', fontsize=28, framealpha=0.8, ncol=2)

###################################################
sns.set_style('dark')
ax4 = plt.subplot(2,2,4)
skanode_to_plot_state = np.load('saved_models/skanode/1/state_test.npy')
sonode_to_plot_state = np.load('saved_models/sonode/1/state_test.npy')
anode_to_plot_state = np.load('saved_models/anode/1/state_test.npy')
rc('font', family='serif')
rc('text', usetex=True)
plt.plot(samp_ts_array, sonode_to_plot_state[...,1], label='SONODE', color='#206491', linestyle='--', lw=3)
plt.plot(samp_ts_array, anode_to_plot_state[...,1], label='ANODE', color='#B39B83', linestyle='--', lw=3)
plt.plot(samp_ts_array, true_state[...,1], label='Ground-truth', color='black', lw=6)
plt.plot(samp_ts_array, skanode_to_plot_state[...,1], label='SKANODE', color='#D15C6B', linestyle='--', lw=4)
plt.xlabel('Time', fontsize=40)
plt.ylabel('Velocity', fontsize=40)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.ylim(-1.1, 1.2)
handles, labels = plt.gca().get_legend_handles_labels()
order = [2,3,0,1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],\
           loc='upper right', fontsize=28, framealpha=0.8, ncol=2)

plt.tight_layout()
plt.savefig('duffing.png', bbox_inches='tight')




names = ['skanode', 's3node', 'sonode', 'anode']
labels =['SKANODE', 'S3NODE', 'SONODE', 'ANODE']

def mse_acc(x):
    acc2_1 = np.load('saved_models/'+names[x]+'/1/acc2_test.npy')
    acc2_2 = np.load('saved_models/'+names[x]+'/2/acc2_test.npy')
    acc2_3 = np.load('saved_models/'+names[x]+'/3/acc2_test.npy')
    
    mse = np.empty(3)
    mse[0] = np.mean((acc2_data[...,0] - acc2_1)**2)
    mse[1] = np.mean((acc2_data[...,0] - acc2_2)**2)
    mse[2] = np.mean((acc2_data[...,0] - acc2_3)**2)
    
    mse_mean = np.mean(mse)
    mse_std = np.std(mse)
    
    print(f"MSE for Acceleration ({labels[x]}) - Mean: {mse_mean:10.6e}, Std: {mse_std:10.6e}")


def mse_dis(x):
    dis_1 = np.load('saved_models/'+names[x]+'/1/state_test.npy')[...,0]
    dis_2 = np.load('saved_models/'+names[x]+'/2/state_test.npy')[...,0]
    dis_3 = np.load('saved_models/'+names[x]+'/3/state_test.npy')[...,0]
    
    mse = np.empty(3)
    mse[0] = np.mean((true_state[...,0] - dis_1)**2)
    mse[1] = np.mean((true_state[...,0] - dis_2)**2)
    mse[2] = np.mean((true_state[...,0] - dis_3)**2)
    
    mse_mean = np.mean(mse)
    mse_std = np.std(mse)
    
    print(f"MSE for Displacement ({labels[x]}) - Mean: {mse_mean:10.6e}, Std: {mse_std:10.6e}")


def mse_vel(x):
    vel_1 = np.load('saved_models/'+names[x]+'/1/state_test.npy')[...,1]
    vel_2 = np.load('saved_models/'+names[x]+'/2/state_test.npy')[...,1]
    vel_3 = np.load('saved_models/'+names[x]+'/3/state_test.npy')[...,1]
    
    mse = np.empty(3)
    mse[0] = np.mean((true_state[...,1] - vel_1)**2)
    mse[1] = np.mean((true_state[...,1] - vel_2)**2)
    mse[2] = np.mean((true_state[...,1] - vel_3)**2)
    
    mse_mean = np.mean(mse)
    mse_std = np.std(mse)
    
    print(f"MSE for Velocity ({labels[x]}) - Mean: {mse_mean:10.6e}, Std: {mse_std:10.6e}")


mse_acc(0)
mse_acc(1)
mse_acc(2)
mse_acc(3)

mse_dis(0)
mse_dis(1)
mse_dis(2)
mse_dis(3)

mse_vel(0)
mse_vel(1)
mse_vel(2)
mse_vel(3)

print()  # Add blank line between MSE and SSIM outputs

def ssim_acc(x):
    acc2_1 = np.load('saved_models/'+names[x]+'/1/acc2_test.npy')
    acc2_2 = np.load('saved_models/'+names[x]+'/2/acc2_test.npy')
    acc2_3 = np.load('saved_models/'+names[x]+'/3/acc2_test.npy')
    
    ssim_vals = np.empty(3)
    ssim_vals[0] = ssim(acc2_data[...,0], acc2_1, data_range=acc2_data[...,0].max() - acc2_data[...,0].min())
    ssim_vals[1] = ssim(acc2_data[...,0], acc2_2, data_range=acc2_data[...,0].max() - acc2_data[...,0].min())
    ssim_vals[2] = ssim(acc2_data[...,0], acc2_3, data_range=acc2_data[...,0].max() - acc2_data[...,0].min())
    
    ssim_mean = np.mean(ssim_vals)
    ssim_std = np.std(ssim_vals)
    
    print(f"SSIM for Acceleration ({labels[x]}) - Mean: {ssim_mean:10.6f}, Std: {ssim_std:10.6f}")


def ssim_dis(x):
    dis_1 = np.load('saved_models/'+names[x]+'/1/state_test.npy')[...,0]
    dis_2 = np.load('saved_models/'+names[x]+'/2/state_test.npy')[...,0]
    dis_3 = np.load('saved_models/'+names[x]+'/3/state_test.npy')[...,0]
    
    ssim_vals = np.empty(3)
    ssim_vals[0] = ssim(true_state[...,0], dis_1, data_range=true_state[...,0].max() - true_state[...,0].min())
    ssim_vals[1] = ssim(true_state[...,0], dis_2, data_range=true_state[...,0].max() - true_state[...,0].min())
    ssim_vals[2] = ssim(true_state[...,0], dis_3, data_range=true_state[...,0].max() - true_state[...,0].min())
    
    ssim_mean = np.mean(ssim_vals)
    ssim_std = np.std(ssim_vals)
    
    print(f"SSIM for Displacement ({labels[x]}) - Mean: {ssim_mean:10.6f}, Std: {ssim_std:10.6f}")


def ssim_vel(x):
    vel_1 = np.load('saved_models/'+names[x]+'/1/state_test.npy')[...,1]
    vel_2 = np.load('saved_models/'+names[x]+'/2/state_test.npy')[...,1]
    vel_3 = np.load('saved_models/'+names[x]+'/3/state_test.npy')[...,1]
    
    ssim_vals = np.empty(3)
    ssim_vals[0] = ssim(true_state[...,1], vel_1, data_range=true_state[...,1].max() - true_state[...,1].min())
    ssim_vals[1] = ssim(true_state[...,1], vel_2, data_range=true_state[...,1].max() - true_state[...,1].min())
    ssim_vals[2] = ssim(true_state[...,1], vel_3, data_range=true_state[...,1].max() - true_state[...,1].min())
    
    ssim_mean = np.mean(ssim_vals)
    ssim_std = np.std(ssim_vals)
    
    print(f"SSIM for Velocity ({labels[x]}) - Mean: {ssim_mean:10.6f}, Std: {ssim_std:10.6f}")


ssim_acc(0)
ssim_acc(1)
ssim_acc(2)
ssim_acc(3)

ssim_dis(0)
ssim_dis(1)
ssim_dis(2)
ssim_dis(3)

ssim_vel(0)
ssim_vel(1)
ssim_vel(2)
ssim_vel(3)