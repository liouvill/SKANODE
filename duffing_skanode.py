import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from kan import *

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-9)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualise', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--npoints', type=int, default=200)
parser.add_argument('--npoints_pred', type=int, default=600)
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--freq', type=float, default=10)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


device = 'cpu'
data = sio.loadmat("duffing_data.mat")

true_state = data['true_latent']
true_state_pred = true_state[0,:args.npoints_pred]

input_force_data = data['force']
observed_data_full = data['obs']

input_force_tensor = torch.tensor(input_force_data[:args.npoints]).float()
true_state_tensor = torch.tensor(true_state[0,:args.npoints]).float()
observed_data = observed_data_full[:args.npoints]
observed_data_pred = observed_data_full[:args.npoints_pred]
observed_tensor = torch.tensor(observed_data).float()
observed_tensor = observed_tensor.reshape(args.npoints, 1).to(device)


def input_force_func(time):
    t = args.freq*time
    if (t > len(input_force_data)-1) or (t < 0):
        return 0
    else:
        t1 = int(math.floor(t))
        delta = t - t1
        if delta == 0:
            return input_force_data[t1]
        else:
            return input_force_data[t1]+delta*(input_force_data[t1+1]-input_force_data[t1])


class KANODEfunc(nn.Module):
    def __init__(self, sys_dim=1, ctrl_dim=1, hidden_dims=[3], grid=5, k=5):
        super(KANODEfunc, self).__init__()
        self.nfe = 0
        input_dim = 2 * sys_dim + ctrl_dim  # [x, v, u]
        output_dim = sys_dim  # dv/dt
        width = [input_dim] + hidden_dims + [output_dim]
        self.kan = KAN.loadckpt('./model/' + '0.12')
        #self.kan = KAN(width=width, grid=grid, k=k)

    def forward(self, t, z):
        self.nfe += 1
        cutoff = int(len(z) // 2)
        x = z[:cutoff]
        v = z[cutoff:]
        t_ = t.detach().cpu().item()
        u = torch.tensor([input_force_func(t_)[0]], dtype=torch.float32).to(z.device)
        z_ = torch.cat((x, v, u)).unsqueeze(0)
        out = self.kan(z_)
        return torch.cat((v, out.squeeze(0)))  # Combine dx/dt = v and dv/dt = f_kan


class Obsfunc(nn.Module):
    def __init__(self, odefunc, nhidden=32):
        super(Obsfunc, self).__init__()
        self.odefunc = odefunc
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        cutoff = int(len(z)/2)
        out = self.odefunc(t, z)
        return out[cutoff:]

class ODEBlock(nn.Module):
    def __init__(self, sys_dim, obs_dim, ctrl_dim):
        super(ODEBlock, self).__init__()
        self.odefunc = KANODEfunc(sys_dim, ctrl_dim)
        self.obsfunc = Obsfunc(self.odefunc)

    def forward(self, integration_times, initial_state=None):
        if initial_state is None:
            z0 = torch.zeros(2, dtype=torch.float).to(device)
        else:
            z0 = torch.tensor(initial_state).float().to(device)
        self.z0 = z0
        state = odeint(self.odefunc, self.z0, integration_times, rtol=args.tol, atol=args.tol, method='rk4')
        T = len(integration_times)
        obs = torch.zeros([T, 1]).to(device)
        for i in range(T):
            obs[i] = self.obsfunc(integration_times[i], state[i])
        return obs, state

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

def safe_plot_kan(kan_model, z_pred, u_vals):
    with torch.no_grad():
        grid_samples = torch.cat([z_pred[:, 0:1], z_pred[:, 1:2], u_vals], dim=1)
        kan_model(grid_samples)
        #kan_model.update_grid_from_samples(grid_samples)
        kan_model.plot(beta=10000, in_vars=['x', r'$\dot{x}$', 'u'], out_vars=['y'])


# ----------------------------
# Training Utilities
# ----------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ----------------------------
# Main Training Loop
# ----------------------------
if __name__ == '__main__':
    filename = f'skanode./{args.experiment_no}./'
    os.makedirs(f'./{filename}', exist_ok=True)

    data_dim = 1
    sys_dim = data_dim
    ctrl_dim = 1
    obs_dim = observed_tensor.shape[-1]

    # Time samples
    samp_ts_array = np.linspace(0., 1 / args.freq * args.npoints, num=args.npoints, endpoint=False)
    samp_ts = torch.tensor(samp_ts_array).float()

    model = ODEBlock(sys_dim, obs_dim, ctrl_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()

    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)
    time_arr = np.empty(args.niters)

    samp_ts_array_pred = np.linspace(0., 1 / args.freq * args.npoints_pred, num=args.npoints_pred, endpoint=False)
    samp_ts_pred = torch.tensor(samp_ts_array_pred).float()

    start_time = time.time()
    min_loss = float('inf')

    for itr in range(1, args.niters + 1):
        model.nfe = 0
        iter_start_time = time.time()

        def closure():
            optimizer.zero_grad()
            pred_obs, _ = model(samp_ts)
            loss = loss_func(pred_obs, observed_tensor)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
        iter_end_time = time.time()

        itr_arr[itr - 1] = itr
        loss_arr[itr - 1] = loss.item()
        nfe_arr[itr - 1] = model.nfe
        time_arr[itr - 1] = iter_end_time - iter_start_time

        print(f'Iter: {itr}, running MSE: {loss.item():.16f}')

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), filename + 'model.pth')

    end_time = time.time()
    
    _, z_pred = model(samp_ts)
    u_vals = torch.tensor([[float(input_force_func(float(t)))] for t in samp_ts], dtype=torch.float32).to(device)
    safe_plot_kan(model.odefunc.kan, z_pred, u_vals)
    
    print('\nTraining complete.')
    print(f'Train MSE = {loss.item()}')
    print(f'NFE = {model.nfe}')
    print(f'Total time = {end_time - start_time}')
    print(f'No. parameters = {count_parameters(model)}')
    print(f'Minimum Loss = {min_loss}')
    formula, var = model.odefunc.kan.symbolic_formula()
    print('Discovered Equation:')
    print(formula[0])
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    np.save(filename+'time_arr.npy', time_arr)


    if args.visualise:
        samp_ts_array_pred = np.linspace(0., 1/args.freq*args.npoints_pred , num = args.npoints_pred, endpoint=False)
        samp_ts_pred = torch.tensor(samp_ts_array_pred).float()
        pred_obs, pred_z = model(samp_ts_pred)
        u_vals = torch.tensor([[float(input_force_func(float(t)))] for t in samp_ts_pred], dtype=torch.float32).to(device)
        to_plot_obs = pred_obs.detach().cpu().numpy().reshape(args.npoints_pred)
        to_plot_state = pred_z.detach().cpu().numpy().reshape(args.npoints_pred,2)
        fig, axs = plt.subplots(3,sharex=True,figsize=(8,6))
        axs[0].plot(samp_ts_array_pred, observed_data_pred, label='True observation')
        axs[0].plot(samp_ts_array_pred, to_plot_obs, label='Predicted observation')
        axs[1].plot(samp_ts_array_pred, true_state_pred[...,0], label='True dis')
        axs[1].plot(samp_ts_array_pred, to_plot_state[...,0], label='Predicted dis')
        axs[2].plot(samp_ts_array_pred, true_state_pred[...,1], label='True vel')
        axs[2].plot(samp_ts_array_pred, to_plot_state[...,1], label='Predicted vel')
        axs[0].set(xlabel='t',ylabel='Acceleration (Obs)')
        axs[1].set(ylabel='Displacement')
        axs[2].set(ylabel='Velocity')
        plt.savefig(filename+'vis.png')
        np.save(filename+'acc2_test.npy', to_plot_obs)
        np.save(filename+'state_test.npy', to_plot_state)
        
        formula, var = model.odefunc.kan.symbolic_formula()
        print(formula[0])
        