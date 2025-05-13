import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm  # or SymLogNorm

from omegaconf import OmegaConf

from SOC_eigf_old2.experiment_settings.settings import define_variables
from SOC_eigf_old2.utils import stochastic_trajectories, control_objective
from SOC_eigf_old2.method import SOC_Solver

device = 'cuda:0'
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

    df = pd.read_csv(f'experiments/{experiment_name}/FBSDE/FBSDE/logs.csv')
    df.dropna(how='all', inplace=True)
    df['run_name'] = 'FBSDE'
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
        fig, ax = plt.subplots(figsize=(5,5))

        index = 'itr'
        col_appendix = ""
        ls = ['-','--','-.']

        for i in range(len(run_names)):
            run_df = df.query(f'run_name=="{run_names[i]}"').copy()
            run_df.drop_duplicates('control_objective_mean',inplace=True)
            run_df.plot(x=index,y='control_objective_mean'+col_appendix, yerr='control_objective_std'+col_appendix, ax = ax, label=f'{labels[run_names[i]]}',capsize=4)
            
            #ax.set_yscale('log')
            ax.set_xlim(0,iters)
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
        ax.legend().set_visible(False)
        plt.savefig(f'figures/{experiment_name}_eigf_objective.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_eigf_objective.png')
    
    if 'grad_log_eigf_error' in df.columns:
        with plt.rc_context({'font.size': 14}):
            fig, ax = plt.subplots(figsize=(5,5))

            index = 'itr'
            col_appendix = "_EMA"

            for i in range(len(run_names)):
                run_df = df.query(f'run_name=="{run_names[i]}"').copy()
                run_df.plot(x=index,y='grad_log_eigf_error'+col_appendix, label=labels[run_names[i]],ax=ax)
            ax.set_yscale('log')
            ax.set_xlim(0,iters)
            ax.grid()
            ax.set_xlabel('Time (s)' if index=="time" else "Iteration")
            ax.set_ylabel(r"$\|\nabla\log \hat\phi_0-\nabla \log\phi_0\|_{\mu}^2$")
            
            #ax.set_ylim(1e-4,10)
            plt.title(title,fontsize=18)
            #plt.legend(loc='upper right', fontsize=14)
            ax.legend().set_visible(False)

            plt.tight_layout()
            plt.savefig(f'figures/{experiment_name}_eigf_gradlog.png', bbox_inches='tight')
            print(f'Successfully saved {experiment_name}_eigf_gradlog.png')
            
            # 2. Grab handles & labels
            handles, labels = plt.gca().get_legend_handles_labels()

            # 3. New figure for the legend
            plt.figure(figsize=(4, 1))               # wider to fit them side by side
            plt.legend(handles, labels,
                    loc='center',                 # middle of the figure
                    ncol=len(labels),             # one column per entry
                    frameon=False)                # optional: remove box
            plt.axis('off')
            plt.tight_layout()

            # 4. Save
            plt.savefig("figures/eigf_legend.png",
                        dpi=300,
                        transparent=True,
                        bbox_inches='tight')
            
def plot_ido(experiment_name, run_names, title = None, not_converged = [], df = None):

    if df is None:
        df = load_ido_df(experiment_name, run_names)

    combined_names = [f'COMBINED/{name}' for name in run_names] 
    run_names = [f'IDO/{name}' for name in run_names] + ['FBSDE']
    
    if experiment_name in ['OU_stable_d20','double_well_d10']:
        run_names += ['rel_GELU']
        df = pd.concat([df,load_eigf_df(experiment_name, ['rel_GELU'])])

    run_names += combined_names

    for run in not_converged:
        run_names.remove(run)
        
    print(run_names)
    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(5,5))

        index = 'itr'
        col_appendix = "_EMA"
        
        colors = []
        ls = []
        for run_name in run_names:
            if 'IDO' in run_name or 'GELU' in run_name or 'FBSDE' in run_name:
                ls += ['-']
            else:
                ls += ['--']
            
            if 'variance' in run_name:
                colors += ['blue']
            elif 'rel_entropy' in run_name:
                colors += ['red']
            elif 'FBSDE' in run_name:
                colors += ['green']
            elif 'SOCM' in run_name:
                colors += ['orange']
            elif 'adjoint_matching' in run_name:
                colors += ['purple']
            elif 'rel_GELU' in run_name:
                colors += ['black']
        
        labels = {
            'FBSDE': 'FBSDE',
            'IDO/rel_entropy': 'Relative entropy',
            'IDO/adjoint_matching': 'Adjoint matching',
            'IDO/SOCM': 'SOCM',
            'IDO/log_variance': 'Log-variance',
            'rel_GELU': 'EIGF (ours)',
            'COMBINED/log_variance': 'EIGF+Log-variance (ours)',
            'COMBINED/rel_entropy': 'EIGF+Relative entropy (ours)',
            'COMBINED/SOCM': 'EIGF+SOCM (ours)',
            'COMBINED/adjoint_matching': 'EIGF+Adjoint Matching (ours)'
        }
        for i in range(len(run_names)):
            if run_names[i] == 'rel_GELU' and experiment_name == 'OU_stable_d20':
                ax.plot([0],[1], label=f'{labels[run_names[i]]}', ls=ls[i],color=colors[i])
            else:
                run_df = df.query(f'run_name=="{run_names[i]}"')
                run_df.plot(x=index,y='control_l2_error'+col_appendix, ax = ax, label=f'{labels[run_names[i]]}', ls=ls[i],color=colors[i])

        ax.set_yscale('log')
        #ax.set_xscale('log')
        ax.grid()
        ax.set_xlabel('Time (s)' if index=="time" else "Iteration")
        ax.set_ylabel('Control $L^2$ error')
        
        #ax.set_ylim(1e-4,10)
        #plt.legend(fontsize=14)
        ax.legend().set_visible(False)
        plt.title(title,fontsize=18)
        plt.tight_layout()
        plt.savefig(f'figures/{experiment_name}_ido_l2err.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_ido_l2err.png')

        if experiment_name == "OU_stable_d20":        
            handles, labels = plt.gca().get_legend_handles_labels()

            ## First row 
            plt.figure(figsize=(4, 1))               # wider to fit them side by side
            plt.legend(handles[:-4], labels[:-4],
                    loc='center',                 # middle of the figure
                    ncol=len(labels[:-4]),             # one column per entry
                    frameon=False)                # optional: remove box
            plt.axis('off')
            plt.tight_layout()

            # 4. Save
            plt.savefig("figures/control_l2_legend_1.png",
                        dpi=300,
                        transparent=True,
                        bbox_inches='tight')
            
            fig, ax = plt.subplots(figsize=(5,5))

            ## Second row 
            plt.figure(figsize=(4, 1))               # wider to fit them side by side
            plt.legend(handles[-4:], labels[-4:],
                    loc='center',                 # middle of the figure
                    ncol=len(labels[-4:]),             # one column per entry
                    frameon=False)                # optional: remove box
            plt.axis('off')
            plt.tight_layout()

            # 4. Save
            plt.savefig("figures/control_l2_legend_2.png",
                        dpi=300,
                        transparent=True,
                        bbox_inches='tight')
            
            fig, ax = plt.subplots(figsize=(5,5))

        index = 'itr'
        col_appendix = ""
        ls = ['-','--','-.']

        # SOCM did not converge, clutters plot
        run_names.remove('IDO/SOCM')

        if experiment_name == "OU_anisotropic_d20":
            run_names.remove('IDO/rel_entropy')

        colors = []
        ls = []
        for run_name in run_names:
            if 'IDO' in run_name or 'GELU' in run_name or 'FBSDE' in run_name:
                ls += ['-']
            else:
                ls += ['--']
            
            if 'variance' in run_name:
                colors += ['blue']
            elif 'rel_entropy' in run_name:
                colors += ['red']
            elif 'FBSDE' in run_name:
                colors += ['green']
            elif 'SOCM' in run_name:
                colors += ['orange']
            elif 'adjoint_matching' in run_name:
                colors += ['purple']
            elif 'rel_GELU' in run_name:
                colors += ['black']

        for i in range(len(run_names)):
            run_df = df.query(f'run_name=="{run_names[i]}"').query('itr>=5000').copy()
            run_df.drop_duplicates('control_objective_mean',inplace=True)
            run_df.plot(x=index,y='control_objective_mean'+col_appendix, yerr='control_objective_std'+col_appendix, ax = ax,capsize=4, color=colors[i])
            
            if experiment_name == "OU_hard_d20":
                ax.set_yscale('log')
                ax.set_ylim(1e2,1e4)
            #ax.set_yscale('log')
            #ax.set_xlim(0,iters)
            ax.grid()
            ax.set_xlabel('Time (s)' if index=="time" else "Iteration")
            ax.set_ylabel('Control objective')
            
        plt.title(title,fontsize=18)
        plt.tight_layout()
        ax.legend().set_visible(False)
        plt.savefig(f'figures/{experiment_name}_objective.png', bbox_inches='tight')
        print(f'Successfully saved {experiment_name}_objective.png')

        

def plot_error_over_time(experiment_name, run_names, title=None, not_converged = [], df = None):
    with plt.rc_context({'font.size': 14}):
        fig, ax = plt.subplots(figsize=(6,6))
        for run in not_converged:
            run_names.remove(run)
        
        run_names.append('FBSDE/FBSDE')
        itrs = [75000]*(len(run_names)-1) + [30000] + [75000]
        labels = run_names[:-1] + ['EIGF']
        
        colors = []
        ls = []

        for run_name in run_names:
            if 'IDO' in run_name or 'EIGF' in run_name or 'FBSDE' in run_name:
                ls += ['-']
            else:
                ls += ['--']
            
            if 'variance' in run_name:
                colors += ['blue']
            elif 'rel_entropy' in run_name:
                colors += ['red']
            elif 'FBSDE' in run_name:
                colors += ['green']
            elif 'SOCM' in run_name:
                colors += ['orange']
            elif 'adjoint_matching' in run_name:
                colors += ['purple']
            elif 'EIGF' in run_name:
                colors += ['black']
            
        for i in range(len(run_names)):
            cfg = OmegaConf.load(f'experiments/{experiment_name}/{run_names[i]}/cfg.yaml')
            cfg.num_steps=200
            cfg.device = device

            cfg.gpu = 5
            ts = torch.linspace(0, cfg.T, cfg.num_steps + 1).to(cfg.device)
            
            x0, sigma, optimal_sde, neural_sde = define_variables(cfg, ts)
            optimal_sde.use_learned_control = False

            state0 = x0.repeat(cfg.optim.batch_size*16, 1)
            
            
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
                    checkpoint = torch.load(f'experiments/{experiment_name}/{run_names[i]}/neural_sde_weights.pth',map_location=device)
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
                solver.load_state_dict(torch.load(f'experiments/{experiment_name}/{run_names[i]}/solver_weights_{itrs[i]:_}.pth',map_location=device),strict=True)
            learned_control = solver.neural_sde.control(ts[:-1],states[:-1]).cpu()
            norm_sqd_diff = torch.sum((target_control - learned_control) ** 2,dim=-1).mean(dim=1)
            norm_sqd_diff_err = torch.std((target_control - learned_control) ** 2,dim=-1).mean(dim=1)
            ax.plot(ts[:-1].cpu(),norm_sqd_diff.detach().cpu(),label=labels[i],color=colors[i],ls=ls[i])

    plt.yscale('log')
    plt.ylim(None,1e2)
    plt.grid()
    ax.legend().set_visible(False)
    plt.xlabel('Time $t$')
    plt.ylabel('Control $L^2$ error')
    plt.title(title,fontsize=18)
    plt.savefig(f'figures/{experiment_name}_l2err_over_time.png', bbox_inches='tight')
    print(f'Successfully saved {experiment_name}_l2err_over_time.png')

def plot_ring():
    with plt.rc_context({'font.size': 14}):
        experiment_name = 'ring_d2/EIGF'
        run_names = ['ritz_GELU','var_GELU','pinn_GELU','rel_GELU']
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

            solver.load_state_dict(torch.load(f'experiments/{experiment_name}/{run_name}/solver_weights_15_000.pth',map_location=device))
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
                'rel_GELU': 'Relative loss (ours)',
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
        print(f'Successfully saved ring_d2_eigf.png')

        # --- parameters ---
        x_lim, y_lim = (-5, 5), (-5, 5)
        bg_res = 200
        heat_cmap = "viridis"
        heat_alpha = 0.6

        # 1) PRE‐COMPUTE ALL Z GRIDS TO FIND GLOBAL vmin/vmax -------------------
        #    This makes two loops, but for bg_res=200 it's still very fast.
        Z_list = []
        xs = torch.linspace(*x_lim, bg_res, device=ts.device)
        ys = torch.linspace(*y_lim, bg_res, device=ts.device)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)

        with torch.no_grad():
            for run in run_names:
                solver = solvers[run]
                Z = -solver.neural_sde.eigf_gs_model(XY).squeeze(-1)
                Z_list.append(Z.cpu().reshape(bg_res, bg_res))

        # compute global min/max
        all_Z = torch.stack([z.flatten() for z in Z_list])

        z_min, z_max = float(all_Z.min()), float(all_Z.max())
        norm = PowerNorm(gamma=0.4, vmin=z_min, vmax=z_max)

        # 2) PLOT EACH SUBPLOT WITH THE SHARED SCALE ---------------------------
        fig, axes = plt.subplots(1, len(run_names), figsize=(5 * len(run_names), 5), sharey=True, sharex=True)
        if len(run_names) == 1:
            axes = [axes]

        for j, run in enumerate(run_names):
            solver = solvers[run]
            Z = Z_list[j]

            # heatmap with common vmin/vmax
            im = axes[j].imshow(
                Z.T,
                extent=[*x_lim, *y_lim],
                origin="lower",
                cmap=heat_cmap,
                alpha=heat_alpha,
                aspect="equal",
                norm=norm, 
                zorder=0,
                label = 'Learned value function'
            )

            # control‐vector quiver (unchanged)
            num_angles = 50
            angles = torch.linspace(0, 2*np.pi, num_angles, device=ts.device)
            rs = torch.linspace(4.5, 5.5, 3, device=ts.device)
            states = torch.cat([
                r/2**0.5*torch.stack([angles.cos(), angles.sin()], dim=1)
                for r in rs
            ])
            controls = solver.control(
                torch.zeros(len(states), device=states.device), states
            )
            x, y = states[:,0].cpu().detach(), states[:,1].cpu().detach()
            u, v = controls[:,0].cpu().detach(), controls[:,1].cpu().detach()

            axes[j].quiver(
                x, y, u, v,
                angles="xy", scale_units="xy", scale=4.0,
                color="black", alpha=1.0, zorder=1
            )

            # labels & styling
            axes[j].set_xlabel("X position")
            axes[j].set_ylabel("Y position")
            axes[j].set_title(labels[run])
            axes[j].set_xlim(*x_lim)
            axes[j].set_ylim(*y_lim)
            axes[j].grid(True)

        # 3) SHARED COLORBAR ----------------------------------------------------
        cbar = fig.colorbar(
            im, ax=axes,
            orientation="vertical",
            fraction=0.02, pad=0.04,
            location="right"
        )
        cbar.set_label("Learned value function $V_0$")  # new label

        plt.suptitle("Learned control inputs and $V_0$")
        #plt.tight_layout()
        plt.savefig(f'figures/ring_d2_visualization.png', bbox_inches='tight')
        print(f'Successfully saved ring_d2_visualization.png')

def plot_exact_eigf():
    torch.random.manual_seed(0)
    cfg = OmegaConf.load('experiment_cfg.yaml')
    cfg.device = "cuda:1"
    cfg.setting = "OU_quadratic_stable"
    cfg.d = 20
    cfg.T = 3.0
    cfg.num_steps=1000
    cfg.optim.batch_size=128*16
    ts = torch.linspace(0, cfg.T, cfg.num_steps + 1).to(cfg.device)
    x0, sigma, optimal_sde, neural_sde = define_variables(cfg, ts)
    optimal_sde.use_learned_control = False

    state0 = x0.repeat(cfg.optim.batch_size, 1)
    states,_,_,_,_,target_controls = stochastic_trajectories(
                                optimal_sde,
                                state0,
                                ts.to(state0),
                                cfg.lmbd,
                                detach=True)
    
    gs_controls = optimal_sde.exact_eigf_control(ts[:-1], states[:-1], 1)

    fig, ax = plt.subplots(figsize=(5,4))
    colormap_name = 'coolwarm'
    colormap = plt.get_cmap(colormap_name)

    max_k = 200
    step = 20
    ks = np.arange(1,max_k,step)
    colors = [colormap(i / (len(ks) - 1)) for i in range(len(ks))]

    # Normalize the color map to the range of ks
    norm = mcolors.Normalize(vmin=ks.min(), vmax=ks.max())
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Needed for the colorbar

    for k in ks:
        color = colormap(norm(k))
        eigf_controls = optimal_sde.exact_eigf_control(ts[:-1], states[:-1], k, verbose=False)
        err = ((eigf_controls - target_controls)**2).sum(dim=2).median(dim=1).values.cpu()
        ax.plot(ts[:-1].cpu(), err, color=color)
        ax.set_yscale('log')

    # Add colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Number of eigenfunctions')
    ax.set_ylabel('Control error $\|u-u^*\|^2$')
    ax.set_xlabel('Time $t$')
    ax.set_title('Control $L^2$ error vs. time $t$\n (truncated eigenfunction solution)')
    ax.grid()
    plt.tight_layout()
    plt.savefig(f'figures/exact_eigf_solution.png', bbox_inches='tight')
    print(f'Successfully saved exact_eigf_solution.png')

def plot_horizon_degradation():

    # Compute for fully converged rel entropy method
    cfg = OmegaConf.load(f'experiments/OU_hard_d20/COMBINED/rel_entropy/cfg.yaml')
    cfg.device = device

    Ts = np.linspace(0.5,5,10)
    steps = Ts*50

    data = {'T': np.zeros(len(Ts)),
            'mean_l2_error': np.zeros(len(Ts)),
            'q05_error': np.zeros(len(Ts)),
            'q95_error': np.zeros(len(Ts))}

    batch_size = 128

    norm_sqd_diff = torch.zeros(1024*cfg.optim.batch_size, device = cfg.device)

    for i in range(len(Ts)):
        with torch.no_grad():
            cfg.num_steps= int(steps[i])
            cfg.T = float(Ts[i])
            cfg.device = device
            cfg.gpu = 5

            ts = torch.linspace(0, cfg.T, cfg.num_steps + 1).to(cfg.device)

            x0, sigma, optimal_sde, neural_sde = define_variables(cfg, ts)
            optimal_sde.use_learned_control = False

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

            solver.load_state_dict(torch.load(f'experiments/OU_hard_d20/COMBINED/rel_entropy/solver_weights_30_000.pth',map_location=device),strict=True)

            for j in range(1024//batch_size):
                state0 = x0.repeat(cfg.optim.batch_size*batch_size, 1)

                torch.random.manual_seed(j)
                states,_,_,_,_,target_control = stochastic_trajectories(
                                            optimal_sde,
                                            state0,
                                            ts.to(state0),
                                            cfg.lmbd,
                                            detach=True)
                target_control = target_control.cpu()
                
                learned_control = solver.neural_sde.control(ts[:-1],states[:-1]).cpu()
                idx1 = j*batch_size*cfg.optim.batch_size
                idx2 = (j+1)*batch_size*cfg.optim.batch_size
                norm_sqd_diff[idx1:idx2] = ((target_control - learned_control) ** 2).sum(dim=-1).mean(dim=0) 
            
            data['T'][i] = Ts[i]
            data['mean_l2_error'][i], data['q05_error'][i], data['q95_error'][i] = norm_sqd_diff.mean(), norm_sqd_diff.quantile(0.05), norm_sqd_diff.quantile(0.95)
        
    combined_df = pd.DataFrame(data)
    combined_df['algo'] = "EIGF+IDO (ours)"

    # Last 1000 iterations for IDO/FBSDE

    experiment_name = "OU_hard_d20_Tscan"

    runs = ['FBSDE']
    runs = ['IDO/adjoint_matching','IDO/relative_entropy','IDO/log_variance','FBSDE']
                        
    run_names = [f'{run}/T={T:.1f}' for run in runs for T in np.linspace(0.5,5.0,10)]

    dfs = []
    for run_name in run_names:
        df = pd.read_csv(f'experiments/{experiment_name}/{run_name}/logs.csv')
        df.dropna(how='all', inplace=True)
        df['run_name'] = run_name
        df['time'] = df['iteration_time'].cumsum()
        df.ffill(inplace=True)
        dfs.append(df)

    df = pd.concat(dfs)
    df['T'] = df['run_name'].str[-3:].astype(float)
    df['algo'] = df['run_name'].str[:-6].replace({
        'IDO/adjoint_matching': 'Adjoint matching (IDO)',
        'IDO/log_variance': 'Log-variance (IDO)',
        'IDO/relative_entropy': 'Relative entropy (IDO)'
        })
    
    # Function to compute mean and 5–95% quantiles for last 1000 iterations
    def last_1000_stats(group):
        last_1000 = group.sort_values('itr')[29000:30000]['control_l2_error']
        return pd.Series({
            'mean_l2_error': last_1000.mean(),
            'q05_error': last_1000.quantile(0.05),
            'q95_error': last_1000.quantile(0.95)
        })

    # Apply the function
    grouped = df.groupby(['algo', 'T']).apply(last_1000_stats).reset_index()

    grouped = pd.concat([grouped, combined_df])

    colors = {'Adjoint matching (IDO)': 'orange',
            'Log-variance (IDO)': 'blue',
            'Relative entropy (IDO)': 'red',
            'FBSDE': 'green',
            'EIGF+IDO (ours)': 'black'}

    # Compute lower and upper error bounds for asymmetric error bars
    grouped['yerr_lower'] = grouped['mean_l2_error'] - grouped['q05_error']
    grouped['yerr_upper'] = grouped['q95_error'] - grouped['mean_l2_error']

    # Plot with asymmetric error bars
    plt.figure(figsize=(5, 5))
    for algo in grouped['algo'].unique():
        subset = grouped[grouped['algo'] == algo]
        plt.errorbar(
            subset['T'], 
            subset['mean_l2_error'], 
            yerr=[subset['yerr_lower'], subset['yerr_upper']],
            label=algo, 
            capsize=4,
            markersize=4, 
            marker='o', 
            linestyle='-',
            color = colors[algo]
        )

    plt.title('Control $L^2$ Error vs $T$ (after 30k iterations)')
    plt.xlabel('Time Horizon $T$')
    plt.ylabel('Control $L^2$ Error')
    plt.yscale('log')
    plt.legend(title='Algorithm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/long_horizon_degradation.png', bbox_inches='tight')
    print(f'Successfully saved long_horizon_degradation.png')

    our_time = pd.read_csv(f'experiments/OU_hard_d20/COMBINED/rel_entropy/logs.csv')[1000:5000]['iteration_time']
    our_df = pd.DataFrame({'algo': 'EIGF+IDO (ours)',
            'mean_l2_error': our_time.mean(),
            'q05_error': our_time.quantile(0.05),
            'q95_error': our_time.quantile(0.95),
            'T': np.linspace(0.5,5,10)})
    
    # Function to compute mean and 5–95% quantiles for last 1000 iterations
    def last_1000_stats(group):
        last_1000 = group.sort_values('itr')[1000:5000]['iteration_time']
        return pd.Series({
            'mean_l2_error': last_1000.mean(),
            'q05_error': last_1000.quantile(0.05),
            'q95_error': last_1000.quantile(0.95)
        })

    # Apply the function
    grouped = df.groupby(['algo', 'T']).apply(last_1000_stats).reset_index()
    grouped = pd.concat([grouped, our_df])
    # Compute lower and upper error bounds for asymmetric error bars
    grouped['yerr_lower'] = grouped['mean_l2_error'] - grouped['q05_error']
    grouped['yerr_upper'] = grouped['q95_error'] - grouped['mean_l2_error']

    # Plot with asymmetric error bars
    plt.figure(figsize=(5,5))
    for algo in grouped['algo'].unique():
        subset = grouped[grouped['algo'] == algo]
        plt.errorbar(
            subset['T'], 
            subset['mean_l2_error'], 
            yerr=[subset['yerr_lower'], subset['yerr_upper']],
            label=algo, 
            capsize=4,
            markersize=4, 
            marker='o', 
            linestyle='-',
            color=colors[algo]
        )

    plt.title('Iteration time vs $T$')
    plt.xlabel('Time Horizon $T$')
    plt.ylabel('Iteration time (s)')
    plt.legend(title='Algorithm')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/long_horizon_iteration_time.png', bbox_inches='tight')
    print(f'Successfully saved long_horizon_iteration_time.png')

    # 2. Grab handles & labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # 3. New figure just for legend
    plt.figure(figsize=(2,1))
    plt.legend(handles, labels, loc='center')
    plt.axis('off')            # hide any axes/frame
    plt.tight_layout()

    # 4. Save it
    plt.savefig("figures/legend_only.png", dpi=300, transparent=True, bbox_inches='tight')

experiment_names = ['OU_stable_d20', 'OU_hard_d20', 'OU_anisotropic_d20', 'double_well_d10']
titles = ['Quadratic (isotropic) (d=20)', 'Quadratic (repulsive) (d=20)', 'Quadratic (anisotropic) (d=20)', 'Double well (d=10)']
iters = [80000,25000,80000,80000]

run_names_list = [['var_GELU','ritz_GELU','pinn_GELU','rel_GELU']] * 4

ido_run_names = ['rel_entropy','log_variance','SOCM','adjoint_matching']

not_converged = [[],['COMBINED/log_variance','COMBINED/adjoint_matching'],['COMBINED/log_variance'],['COMBINED/log_variance','COMBINED/adjoint_matching','COMBINED/SOCM','COMBINED/rel_entropy']]
#plot_ring()
for i in range(len(experiment_names)):
    #plot_eigf(experiment_names[i], run_names_list[i], iters=iters[i],title=titles[i])

    plot_ido(experiment_names[i], ido_run_names, titles[i], not_converged[i])

    over_time_names = [f'IDO/{ido_run}' for ido_run in ido_run_names] + [f'COMBINED/{ido_run}' for ido_run in ido_run_names] + ['EIGF/rel_GELU']
    
    #plot_error_over_time(experiment_names[i],over_time_names,titles[i], not_converged[i])

    pass

#plot_eigf('ring_d2',run_names_list[0],iters=20000,title="Ring (d=2)")

#plot_exact_eigf()
#plot_horizon_degradation()