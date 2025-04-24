# A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control
This (branch of the) repository contains the code used for the master thesis "A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control" at ETH Zurich. The goal of the work is to design a numerical solver for stochastic optimal control (SOC) problems by using neural networks to learn the eigenfunctions of an associated Schrödinger operator.

## User guide

To install the required libraries, run `pip install requirements.txt`. 

The folder `SOC_eigf` contains the code of the method, and the file `main.py` can be used to run experiments. The configuration of the experiment should be specified in the `experiment_cfg.yaml` file or otherwise specified using flags. To run multiple experiments in parallel, pass a list as one or more of the arguments. For instace, running 

<pre>
```
torchrun main.py --master_port=29500 setting="double_well" d=10 method="EIGF" solver.eigf_loss=["ritz","var"] gpu=[0,1] run_name=["ritz_test","var_test"] experiment_name="double_well_d10"
```
</pre>

will run the eigenfunction method for the `DoubleWell` setting in $d=10$, using the Ritz loss on `cuda:0` and the variational loss on `cuda:1`, and save the results in `experiments/double_well_d10/EIGF/ritz_test` and `experiments/double_well_d10/EIGF/var_test` respectively.

## Reproducing experiments

The file `exp_cmds.txt` contains the commands used to perform the experiments documented in the thesis. Running these commands creates a folder for the experiment and a file `logs.csv` with relevant metrics during the run. It also saves checkpoints of the model weights as `solver_weights.pth`. 

The code used to analyze the results and generate plots is given in `plots.ipynb`. In addition, the script `create_plots.py` generates plots of the control objective, L2 error and eigenfunction metrics for each experiment listed in `experiments_list.txt` in the directory `figures`.

For estimating the computation cost, the bash file `time_experiments.sh` reruns experiments found in `experiments_list.txt` sequentially for 1000 iterations, saving the results in the folder `timing_experiments`. The notebook `timing.ipynb` contains code to analyze the average time per iteration from this data.