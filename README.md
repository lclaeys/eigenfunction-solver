# A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control
This repository contains the code used for the paper "A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control". The goal of the work is to design a numerical solver for stochastic optimal control (SOC) problems by using neural networks to learn the eigenfunctions of an associated Schrödinger operator.

## User guide

To install the required libraries, run `pip install -r requirements.txt`. 

The folder `SOC_eigf` contains the code of the method, and the file `main.py` can be used to run experiments. The configuration of the experiment should be specified in the `experiment_cfg.yaml` file or otherwise specified using flags. To run multiple experiments in parallel, pass a list as one or more of the arguments. For example, running 

```
torchrun --master_port=29500 main.py setting="double_well" d=10 method="EIGF" solver.eigf_loss=["ritz","rel"] gpu=[0,1] run_name=["ritz_test","rel_test"] experiment_name="double_well_d10"
```

will run the eigenfunction method for the `DoubleWell` setting in $d=10$, using the deep Ritz loss on `cuda:0` and the relative loss on `cuda:1`, and save the results in `experiments/double_well_d10/EIGF/ritz_test` and `experiments/double_well_d10/EIGF/rel_test` respectively.

## Reproducing experiments

The file `exp_cmds.txt` contains the commands used to perform the experiments documented in the paper. Running these commands creates a folder for the experiment and a file `logs.csv` with relevant metrics saved during the run. It also saves checkpoints of the model weights as `solver_weights_{itr}.pth`. 

The code used to analyze the results and generate plots is given in `plots.ipynb`. In addition, the script `create_plots.py` generates plots of the control objective, average L2 error as a function of iteration, L2 error as a function of iteration time, and the performance of different eigenfunction losses for each experiment listed in `experiments_list.txt` in the directory `figures`. It also creates the other figures in the paper (performance deterioration with increasing $T$, ring example etc.) and estimates the objective, printing to the file `objectives.txt`. The script can be edited to only generate a subset of these figures.

For estimating the computation cost, the bash file `time_experiments.sh` reruns experiments found in `experiments_list.txt` sequentially for 1000 iterations, saving the results in the folder `timing_experiments`. The notebook `timing.ipynb` contains code to analyze the average time per iteration from this data.

## Config file documentation

We give an overview of all of the parameters that can be specified in `experiment_cfg.yaml`:

| Parameter                         | Description                                | Example/default value   |
|----------------------------------|--------------------------------------------|-----------------|
| `setting`                        | Experiment setup name                      | `double_well_d10`   |
| `d`                              | Dimensionality of the system               | `10`            |
| `device`                         | Computation device                         | `cuda`          |
| `gpu`                            | GPU index to use                           | `0`             |
| `lmbd`                           | Noise level $\lambda = \beta^{-1}$         | `1.0`           |
| `T`                              | Time horizon                               | `4.0`           |
| `eval_frac`                      | Evaluate on $[0, aT]$ (debugging)          | `1.0`           |
| `num_steps`                      | Number of steps in simulation              | `400`           |
| `method`                         | Training method used (`EIGF`, `COMBINED`,`IDO`, `FBSDE`)| `IDO`      |
| `experiment_name`                | Name of the experiment folder              | `double_well_d10` |
| `run_name`                       | Name of the run                            | `SOCM`  |
| `seed`                           | Random seed for reproducibility            | `0`             |
| `num_iterations`                 | Total training iterations                  | `80000`         |
| `save_model_every`               | Save model every N iterations              | `5000`          |
| `log_every`                      | Log training metrics every N iterations         | `100`           |
| `compute_control_error_every`   | Compute control error every N iterations        | `100`           |
| `compute_objective_every`       | Compute objective function every N iterations   | `1000`          |
| `objective_samples`             | Number of samples for objective computation | `65536`         |
| `trained_eigf_run_name`         | Run name used for pretrained eigenfunctions (in combined method) | `rel_GAUSS`     |
| `use_exact_eigvals`              | Whether to use known explicit eigenvalues (debugging) | `false`|
| `eigf.k`                         | Number of eigenfunctions to train          | `2`             |
| `eigf.hdims`                     | Hidden layers for eigenfunction network    | `[256, 256, 256]` |
| `eigf.arch`                      | Architecture used for eigenfunction        | `GAUSS`         |
| `ido.hdims`                      | Hidden layers for IDO network              | `[256, 128, 64]` |
| `ido.hdims_M`                    | Hidden layers for M-network in SOCM method | `[128, 128]`    |
| `ido.gamma`                      | Gamma parameter                            | `1.0`           |
| `ido.gamma1`                     | Gamma1 parameter                           | `1.0`           |
| `ido.gamma2`                     | Gamma2 parameter                           | `1.0`           |
| `ido.scaling_factor_nabla_V`    | Initialization scale for IDO method         | `1.0`           |
| `ido.scaling_factor_M`          | Initialization scale for M-network in SOCM | `1.0`           |
| `ido.T_cutoff`                  | Cutoff time for combined method               | `1.0`           |
| `solver.langevin_burnin_steps`  | Burn-in steps for Langevin sampling        | `1000`          |
| `solver.langevin_sample_steps`  | Sample steps for Langevin                  | `100`           |
| `solver.langevin_dt`            | Time step for Langevin dynamics            | `0.01`          |
| `solver.beta`                   | regularization parameter for solver        | `0.15`          |
| `solver.eigf_loss`              | Loss type for eigenfunctions               | `ritz`          |
| `solver.finetune`               | Train with Ritz loss until first eigenvalue converged | `true`          |
| `solver.nsamples`               | Batch size for eigenfunction learning           | `65536`         |
| `solver.ido_algorithm`          | IDO algorithm variant used                 | `SOCM`  |
| `solver.fbsde_reg`              | Regularizer for FBSDE algorithm             | `0.0`           |
| `solver.ritz_steps`             | Minimum number of deep Ritz steps before using non-Ritz loss | `5000` |
| `optim.batch_size`              | Batch size for optimization in IDO/COMBINED| `64`            |
| `optim.adam_lr`                 | Learning rate for EIGF/COMBINED/FBSDE            | `0.0001`        |
| `optim.adam_eps`                | Epsilon value for Adam                     | `1.0e-08`       |
| `optim.adam_wd`                 | Weight decay for Adam                      | `1.0e-08`       |
| `optim.ido_lr`                  | Learning rate for IDO                      | `0.0001`        |
| `optim.M_lr`                    | Learning rate for M-net                    | `0.01`          |
| `optim.y0_lr`                   | Learning rate for initial state            | `0.01`          |
| `delta_t_optimal`               | Time discretization for finite difference  | `0.0001`        |
| `delta_x_optimal`               | Space discretization for finite difference | `0.01`          |
| `timing`                        | Flag to enable timing metrics              | `false`         |
