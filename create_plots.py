import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from omegaconf import OmegaConf

from SOC_eigf.experiment_settings.settings import define_variables
from SOC_eigf.utils import stochastic_trajectories, control_objective
from SOC_eigf.method import SOC_Solver

def load_eigf_df(experiment_name, run_names):
    dfs = []
    for run_name in run_names:
        df = pd.read_csv(f'experiments/{experiment_name}/EIGF/{run_name}/logs.csv')
        df.dropna(how='all', inplace=True)
        df['run_name'] = run_name
        df['time'] = df['iteration_time'].cumsum()
        df.ffill(inplace=True)
        dfs.append(df)

    df = pd.concat(dfs)

    EMA_halflife = 200
    for column in df.columns:
        if 'error' in column or 'loss' in column:
            df[f'{column}_EMA'] = (
                df.groupby('run_name')[column]
                #.transform(lambda x: np.exp(x.apply(lambda x: np.log(x)).ewm(halflife=EMA_halflife, adjust=False).mean()))
                .transform(lambda x: x.ewm(halflife=EMA_halflife, adjust=False).mean())
            )
    
    return df

def load_ido_df(experiment_name, run_names):
    dfs = []
    for run_name in run_names:
        df = pd.read_csv(f'experiments/{experiment_name}/IDO/{run_name}/logs.csv')
        df.dropna(how='all', inplace=True)
        df['run_name'] = 'IDO/' + run_name
        df['time'] = df['iteration_time'].cumsum()
        df.ffill(inplace=True)
        dfs.append(df)

        df = pd.read_csv(f'experiments/{experiment_name}/COMBINED/{run_name}/logs.csv')
        df.dropna(how='all', inplace=True)
        df['run_name'] = 'COMBINED/' + run_name
        df['time'] = df['iteration_time'].cumsum()
        df.ffill(inplace=True)
        dfs.append(df)

    df = pd.concat(dfs)

    EMA_halflife = 200
    for column in df.columns:
        if 'error' in column or 'loss' in column:
            df[f'{column}_EMA'] = (
                df.groupby('run_name')[column]
                #.transform(lambda x: np.exp(x.apply(lambda x: np.log(x)).ewm(halflife=EMA_halflife, adjust=False).mean()))
                .transform(lambda x: x.ewm(halflife=EMA_halflife, adjust=False).mean())
            )

    return df

def plot_eigf(experiment_name, run_names, df = None, iters=None):
    if df is None:
        df = load_eigf_df(experiment_name, run_names)
    
    with plt.rc_context({'font.size': 12}):
        fig, axes = plt.subplots(1, len(run_names), figsize=(4.5*len(run_names),6),sharey=True,sharex=True)  # 1 row, 2 columns

        index = 'itr'
        col_appendix = "_EMA"
        labels = {
            'var_GELU': r'$\beta\langle f, \mathcal{L}f\rangle_\mu + \langle f, f\rangle_\mu$ (Var loss)',
            'ritz_GELU': r'$\frac{\langle f, \mathcal{L}f\rangle_\mu}{\langle f, f\rangle_\mu}$ (Deep Ritz loss)',
            'var_GAUSS': r'$\beta\langle f, \mathcal{L}f\rangle_\mu + \langle f, f\rangle_\mu$ (Var loss)',
            'ritz_GAUSS': r'$\frac{\langle f, \mathcal{L}f\rangle_\mu}{\langle f, f\rangle_\mu}$ (Deep Ritz loss)',
            'ritz': r'$\frac{\langle f, \mathcal{L}f\rangle_\mu}{\langle f, f\rangle_\mu}$ (Deep Ritz loss)',
            'pinn_GAUSS': r'$\|\mathcal{L}f-\lambda f\|_\mu^2$ (PINN loss)',
            'rel_GAUSS': r'$\|\mathcal{L}f/f-\lambda\|_\mu^2$ (Relative loss)',
            'log_rel_GAUSS': r'$\|\log (\mathcal{L}f/\lambda f)\|_\mu^2$ (Log relative loss)'
        }
        running_min = 1.0
        for i in range(len(run_names)):
            run_df = df.query(f'run_name=="{run_names[i]}"')
            run_df.plot(x=index,y='loss'+col_appendix, color='black', ax = axes[i],label='Loss')
            if 'eigf_error' in run_df.columns:
                run_df.plot(x=index,y='eigf_error'+col_appendix, color='blue', ax = axes[i],label=r"$\|f-\phi\|_{\mu}^2$")
                run_df.plot(x=index,y='grad_log_eigf_error'+col_appendix, color='red', ax = axes[i],label=r"$\|\nabla\log f-\nabla \log\phi\|_{\mu}^2$")
            run_df.plot(x=index,y='control_l2_error'+col_appendix, color='green', ax = axes[i],label=r"Control $L^2$ error")

            #ax[i].set_title("$\|f-\phi\|_{\mu}^2$")
            axes[i].set_title(labels[run_names[i]],fontsize=18)
            axes[i].set_yscale('log')
            axes[i].grid()
            axes[i].set_xlabel('Time (s)' if index=="time" else "Iteration")
            axes[i].set_xlim(0,iters)

        for ax in axes:
            ax.legend().set_visible(False)

        axes[0].set_ylim(None,1e1)
        # Create shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.12))
        plt.suptitle('Loss and error (EMA) for different loss functions',fontsize=22)

        plt.tight_layout()
        plt.savefig(f'figures/{experiment_name}_eigf.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_eigf.png')

    with plt.rc_context({'font.size': 12}):
        fig, ax = plt.subplots(figsize=(12,6))

        index = 'itr'
        col_appendix = ""
        ls = ['-','--','-.']

        for i in range(len(run_names)):
            run_df = df.query(f'run_name=="{run_names[i]}"').copy()
            run_df.drop_duplicates('control_objective_mean',inplace=True)
            run_df.plot(x=index,y='control_objective_mean'+col_appendix, yerr='control_objective_std'+col_appendix, ax = ax, label=f'{run_names[i]}',capsize=4)
            
            #ax.set_yscale('log')
            #ax.set_xlim(100,)
            ax.grid()
            ax.set_xlabel('Time (s)' if index=="time" else "Iteration")
            ax.set_ylabel('Control objective')
        
        #ax.set_ylim(1e-4,10)
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(f'figures/{experiment_name}_eigf_objective.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_eigf_objective.png')

def plot_ido(experiment_name, run_names, df = None):
    if df is None:
        df = load_ido_df(experiment_name, run_names)

    run_names = [f'IDO/{name}' for name in run_names] + [f'COMBINED/{name}' for name in run_names]
    with plt.rc_context({'font.size': 12}):
        fig, ax = plt.subplots(figsize=(12,6))

        index = 'itr'
        col_appendix = "_EMA"
        
        colors = []
        ls = []
        for run_name in run_names:
            if 'IDO' in run_name or 'EIGF' in run_name:
                ls += ['-']
            else:
                ls += ['-.']
            
            if 'variance' in run_name:
                colors += ['blue']
            elif 'rel_entropy' in run_name:
                colors += ['red']
            elif 'SOCM_adjoint' in run_name:
                colors += ['green']
            elif 'SOCM' in run_name:
                colors += ['orange']
            elif 'EIGF' in run_name:
                colors += ['black']

        for i in range(len(run_names)):
            run_df = df.query(f'run_name=="{run_names[i]}"')
            run_df.plot(x=index,y='control_l2_error'+col_appendix, ax = ax, label=f'{run_names[i]}', ls=ls[i],color=colors[i])

            ax.set_yscale('log')
            #ax.set_xscale('log')
            ax.grid()
            ax.set_xlabel('Time (s)' if index=="time" else "Iteration")
            ax.set_ylabel('Control $L^2$ error')
        
        #ax.set_ylim(1e-4,10)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f'figures/{experiment_name}_ido_l2err.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_ido_l2err.png')

experiment_names = ['OU_stable_d20',  'OU_hard_d20', 'OU_anisotropic_d20', 'double_well_d10']
iters = [40000,40000,40000,30000]

run_names_list = [['var_GELU','ritz_GELU','pinn_GAUSS','rel_GAUSS']] * 2 + [['var_GAUSS','ritz_GAUSS','pinn_GAUSS','rel_GAUSS']] * 2

ido_run_names = ['rel_entropy','log_variance','SOCM','SOCM_adjoint']

for i in range(len(experiment_names)):
    plot_eigf(experiment_names[i], run_names_list[i], iters=iters[i])

    plot_ido(experiment_names[i], ido_run_names)


