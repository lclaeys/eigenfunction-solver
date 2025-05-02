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

def plot_eigf(experiment_name, run_names, title=None, df = None, iters=None):
    if df is None:
        df = load_eigf_df(experiment_name, run_names)
    
    if "control_l2_error" in df.columns:
        with plt.rc_context({'font.size': 14}):
            fig, axes = plt.subplots(1, len(run_names), figsize=(4.5*len(run_names),6),sharey=True,sharex=True)  # 1 row, 2 columns

            index = 'itr'
            col_appendix = "_EMA"
            running_min = 1.0
            for i in range(len(run_names)):
                run_df = df.query(f'run_name=="{run_names[i]}"')
                run_df.plot(x=index,y='loss'+col_appendix, color='black', ax = axes[i],label='Loss')
                if 'eigf_error' in run_df.columns:
                    run_df.plot(x=index,y='eigf_error'+col_appendix, color='blue', ax = axes[i],label=r"$\|f-\phi\|_{\mu}^2$")
                    run_df.plot(x=index,y='grad_log_eigf_error'+col_appendix, color='red', ax = axes[i],label=r"$\|\nabla\log f-\nabla \log\phi\|_{\mu}^2$")
                run_df.plot(x=index,y='control_l2_error'+col_appendix, color='green', ax = axes[i],label=r"Control $L^2$ error")

                #ax[i].set_title("$\|f-\phi\|_{\mu}^2$")
                axes[i].set_title(run_names[i],fontsize=18)
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

    labels = {
        'rel_GELU': 'Relative loss',
        'var_GELU': 'Variational loss',
        'pinn_GELU': 'PINN loss',
        'ritz_GELU': 'Deep Ritz loss'
    }

    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(6,6))

        index = 'itr'
        col_appendix = ""
        ls = ['-','--','-.']

        for i in range(len(run_names)):
            run_df = df.query(f'run_name=="{run_names[i]}"').copy()
            run_df.drop_duplicates('control_objective_mean',inplace=True)
            run_df.plot(x=index,y='control_objective_mean'+col_appendix, yerr='control_objective_std'+col_appendix, ax = ax, label=f'{labels[run_names[i]]}',capsize=4)
            
            #ax.set_yscale('log')
            #ax.set_xlim(100,)
            ax.grid()
            ax.set_xlabel('Time (s)' if index=="time" else "Iteration")
            ax.set_ylabel('Control objective')
        
        if experiment_name == "double_well_d10":
            ax.set_ylim(32,33)
            plt.legend()
        else:
            plt.legend(loc='center right')
        
        plt.title(title,fontsize=18)
        plt.tight_layout()
        plt.savefig(f'figures/{experiment_name}_eigf_objective.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_eigf_objective.png')
    
    if 'grad_log_eigf_error' in df.columns:
        with plt.rc_context({'font.size': 14}):
            fig, ax = plt.subplots(figsize=(6,6))

            index = 'itr'
            col_appendix = "_EMA"

            for i in range(len(run_names)):
                run_df = df.query(f'run_name=="{run_names[i]}"').copy()
                run_df.plot(x=index,y='grad_log_eigf_error'+col_appendix, label=labels[run_names[i]],ax=ax)
            ax.set_yscale('log')
            #ax.set_xlim(100,)
            ax.grid()
            ax.set_xlabel('Time (s)' if index=="time" else "Iteration")
            ax.set_ylabel(r"$\|\nabla\log \hat\phi_0-\nabla \log\phi_0\|_{\mu}^2$")
            
            #ax.set_ylim(1e-4,10)
            plt.title(title,fontsize=18)
            plt.legend(loc='upper right', fontsize=14)

            plt.tight_layout()
            plt.savefig(f'figures/{experiment_name}_eigf_gradlog.png', bbox_inches='tight')
            print(f'Successfully saved {experiment_name}_eigf_gradlog.png')

def plot_ido(experiment_name, run_names, title = None, df = None):
    if df is None:
        df = load_ido_df(experiment_name, run_names)

    run_names = [f'IDO/{name}' for name in run_names] + [f'COMBINED/{name}' for name in run_names]
    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(6,6))

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
                colors += ['orange']
            elif 'SOCM' in run_name:
                colors += ['green']
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
        plt.legend(fontsize=14)
        plt.title(title,fontsize=18)
        plt.tight_layout()
        plt.savefig(f'figures/{experiment_name}_ido_l2err.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_ido_l2err.png')

def plot_error_over_time(experiment_name, run_names, title=None,df = None):
    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(6,6))
        itrs = [75000]*(len(run_names)-1) + [75000]
        labels = run_names[:-1] + ['EIGF']
        for i in range(len(run_names)):
            cfg = OmegaConf.load(f'experiments/{experiment_name}/{run_names[i]}/cfg.yaml')
            cfg.num_steps=200
            cfg.device = 'cuda:5'

            cfg.gpu = 5
            ts = torch.linspace(0, cfg.T, cfg.num_steps + 1).to(cfg.device)
            
            x0, sigma, optimal_sde, neural_sde = define_variables(cfg, ts)
            optimal_sde.use_learned_control = False

            state0 = x0.repeat(cfg.optim.batch_size*16, 1)
            
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
                    colors += ['orange']
                elif 'SOCM' in run_name:
                    colors += ['green']
                elif 'EIGF' in run_name:
                    colors += ['black']

            if i == 0:
                states,_,_,_,_,target_control = stochastic_trajectories(
                                            optimal_sde,
                                            state0,
                                            ts.to(state0),
                                            cfg.lmbd,
                                            detach=True)
                target_control = target_control.cpu()
            if cfg.method == "EIGF":
                try:
                    checkpoint = torch.load(f'experiments/{experiment_name}/{run_names[i]}/neural_sde_weights.pth',map_location='cuda:5')
                except FileNotFoundError as e:
                    print('Error: Please train EIGF model before attempting to train combined model!')
                    raise e
                
                neural_sde.load_state_dict(checkpoint, strict=False)
            solver = SOC_Solver(
                    neural_sde,
                    x0,
                    None,
                    T=cfg.T,
                    num_steps=cfg.num_steps,
                    lmbd=cfg.lmbd,
                    d=cfg.d,
                    sigma=sigma,
                    solver_cfg=cfg.solver
                )
            if cfg.method != "EIGF":
                solver.load_state_dict(torch.load(f'experiments/{experiment_name}/{run_names[i]}/solver_weights_{itrs[i]:_}.pth',map_location='cuda:5'),strict=True)
            learned_control = solver.neural_sde.control(ts[:-1],states[:-1]).cpu()
            norm_sqd_diff = torch.sum((target_control - learned_control) ** 2,dim=-1).mean(dim=1)
            norm_sqd_diff_err = torch.std((target_control - learned_control) ** 2,dim=-1).mean(dim=1)
            ax.plot(ts[:-1].cpu(),norm_sqd_diff.detach().cpu(),label=labels[i],color=colors[i],ls=ls[i])

    plt.yscale('log')
    plt.ylim(None,1e2)
    plt.grid()
    plt.legend(loc='upper left',fontsize=14)
    plt.xlabel('Time $t$')
    plt.ylabel('Control $L^2$ error')
    plt.title(title,fontsize=18)
    plt.savefig(f'figures/{experiment_name}_l2err_over_time.png', bbox_inches='tight')
    print(f'Successfully saved {experiment_name}_l2err_over_time.png')

def plot_ring():
    with plt.rc_context({'font.size': 14}):
        experiment_name = 'ring_d2/EIGF'
        run_names = ['ritz_GELU','pinn_GELU','rel_GELU']
        solvers = {}
        for run_name in run_names:
            cfg = OmegaConf.load(f'experiments/{experiment_name}/{run_name}/cfg.yaml')
            cfg.device = "cuda:7"
            cfg.num_steps=500
            ts = torch.linspace(0, cfg.T, cfg.num_steps + 1).to(cfg.device)

            x0, sigma, optimal_sde, neural_sde = define_variables(cfg, ts)

            state0 = x0.repeat(cfg.optim.batch_size*16, 1)

            solver = SOC_Solver(
                    neural_sde,
                    x0,
                    None,
                    T=cfg.T,
                    num_steps=cfg.num_steps,
                    lmbd=cfg.lmbd,
                    d=cfg.d,
                    sigma=sigma,
                    solver_cfg=cfg.solver
                )

            solver.load_state_dict(torch.load(f'experiments/{experiment_name}/{run_name}/solver_weights_15_000.pth',map_location='cuda:7'))
            solvers[run_name] = solver

        dfs = []
        for run_name in run_names:
            df = pd.read_csv(f'experiments/{experiment_name}/{run_name}/logs.csv')
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
                    .transform(lambda x: np.exp(x.apply(lambda x: np.log(x)).ewm(halflife=EMA_halflife, adjust=False).mean()))
                    #.transform(lambda x: x.apply(lambda x: x.ewm(halflife=EMA_halflife, adjust=False).mean()))
                )

        print(df.columns)

        labels = {'pinn_GELU': 'PINN loss',
                'rel_GELU': 'Relative loss',
                'var_GELU': 'Variational loss',
                'ritz_GELU': 'Deep Ritz loss'}
        
        fig, axes = plt.subplots(1,len(run_names),figsize=(5*len(run_names),5)) 

        colors = ['red','blue']
        if len(run_names) == 1:
            axes = [axes]
        for j in range(len(run_names)):
            solver = solvers[run_names[j]]
            num = 50
            angles = torch.linspace(0,2*np.pi,num,device=ts.device)
            r_states = []
            n = 3
            rs = torch.linspace(4.5,5.5,n)
            for r in rs:
                r_states.append(r/2**(1/2) * torch.stack([angles.cos(),angles.sin()],dim=1))

            states = torch.concatenate(r_states)
            controls = solver.control(torch.zeros(len(states),device=states.device), states)[:-1,:]
            states= states.unsqueeze(1)
            controls = controls.unsqueeze(1)

            states = states[:-1].cpu().detach()
            controls = controls.cpu().detach()

            i = 0
            dt = (ts[1] - ts[0]).cpu()
            # Extract the first batch for visualization
            x, y = states[:, i, 0], states[:, i, 1]
            u, v = controls[:, i, 0], controls[:, i, 1]

            colors = cm.viridis(np.linspace(0, 1, len(x)))
                
            # Plot control vectors as arrows
            axes[j].quiver(x, y, u, v, angles='xy', scale_units='xy', color=colors[j], alpha=1.0, scale=4.0, label="Controls")

            # Labels and title
            axes[j].set_xlabel("X Position")
            axes[j].set_ylabel("Y Position")
            axes[j].set_title(labels[run_names[j]])

            # Ensure equal aspect ratio
            #ax.set_aspect('equal', adjustable='box')
            axes[j].set_xlim(-5,5)
            axes[j].set_ylim(-5,5)

            #axes[j].legend()
            axes[j].grid()
        #plt.suptitle('Control inputs')
        plt.savefig(f'figures/ring_d2_controls.png', bbox_inches='tight')

        # --- 1) build your common grid in advance ---
        r, delta = 2.5, 0.5
        n_rho, n_theta = 100, 200

        rho = torch.linspace(r - delta, r + delta, n_rho, device=cfg.device)
        theta = torch.linspace(0, 2 * np.pi, n_theta, device=cfg.device)
        RHO, THETA = torch.meshgrid(rho, theta)

        X = RHO * torch.cos(THETA)
        Y = RHO * torch.sin(THETA)
        XY = torch.stack([X.ravel(), Y.ravel()], dim=1)

        X_np = X.cpu().numpy()
        Y_np = Y.cpu().numpy()

        # --- 2) compute global mins/maxs ---
        raw_min, raw_max = float('inf'), float('-inf')
        log_min, log_max = float('inf'), float('-inf')

        Zs_raw, Zs_log = [], []
        for run_name in run_names:
            solver = solvers[run_name]
            f = lambda pts: solver.neural_sde.eigf_gs_model(pts)
            F = (f(XY).reshape(X.shape).detach().cpu().numpy())
            Z_raw = np.exp(F)
            Z_log = F
            Zs_raw.append(Z_raw)
            Zs_log.append(Z_log)
            raw_min, raw_max = min(raw_min, Z_raw.min()), max(raw_max, Z_raw.max())
            log_min, log_max = min(log_min, Z_log.min()), max(log_max, Z_log.max())

        # --- 3) plotting with shared limits, bigger fonts, no z‑label ---
        n_runs = len(run_names)
        fig, axes = plt.subplots(2, n_runs,
                                subplot_kw={'projection': '3d'},
                                figsize=(5 * n_runs,10))


        for j, run_name in enumerate(run_names):
            if n_runs == 1:
                ax1, ax2 = axes[0], axes[1]
            else:
                ax1, ax2 = axes[0,j], axes[1,j]
            Z_raw, Z_log = Zs_raw[j], Zs_log[j]

            # --- raw eigenfunction ---
            surf1 = ax1.plot_surface(X_np, Y_np, Z_raw, cmap='viridis')
            ax1.set_xlim(X_np.min(), X_np.max())
            ax1.set_ylim(Y_np.min(), Y_np.max())
            ax1.set_zlim(raw_min, raw_max)
            ax1.set_xlabel('x', fontsize=14)
            ax1.set_ylabel('y', fontsize=14)
            # remove or hide z‑axis label
            ax1.zaxis.label.set_visible(False)
            ax1.set_title(f'{labels[run_name]} — eigenfunction', fontsize=16)
            ax1.tick_params(labelsize=12)
            ax1.view_init(elev=30, azim=225)

            # --- log eigenfunction ---
            surf2 = ax2.plot_surface(X_np, Y_np, Z_log, cmap='viridis')
            ax2.set_xlim(X_np.min(), X_np.max())
            ax2.set_ylim(Y_np.min(), Y_np.max())
            ax2.set_zlim(log_min, log_max)
            ax2.set_xlabel('x', fontsize=14)
            ax2.set_ylabel('y', fontsize=14)
            ax2.zaxis.label.set_visible(False)
            ax2.set_title(f'{labels[run_name]} — log eigenfunction', fontsize=16)
            ax2.tick_params(labelsize=12)
            ax2.view_init(elev=30, azim=225)

        plt.tight_layout()
        plt.savefig(f'figures/ring_d2_eigf.png', bbox_inches='tight')

experiment_names = ['OU_stable_d20', 'OU_anisotropic_d20', 'double_well_d10']
titles = ['Quadratic (isotropic) (d=20)', 'Quadratic (anisotropic) (d=20)', 'Double well (d=10)']
iters = [80000,80000,80000]

run_names_list = [['var_GELU','ritz_GELU','pinn_GELU','rel_GELU']] * 3

ido_run_names = ['rel_entropy','log_variance','SOCM']

plot_ring()
for i in range(len(experiment_names)):
    plot_eigf(experiment_names[i], run_names_list[i], iters=iters[i],title=titles[i])

    plot_ido(experiment_names[i], ido_run_names, titles[i])

    over_time_names = [f'IDO/{ido_run}' for ido_run in ido_run_names] + [f'COMBINED/{ido_run}' for ido_run in ido_run_names] + ['EIGF/rel_GELU']
    
    if i < 2:
        plot_error_over_time(experiment_names[i],over_time_names,titles[i])

plot_eigf('ring_d2',run_names_list[0],iters=20000,title="Ring (d=2)")

