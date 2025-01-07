# eigenfunction-solver
Given a potential $V$, we want to compute the eigenfunctions and eigenvalues of $L = -\Delta + \langle \nabla V, \nabla \cdot\rangle$
in order to solve the PDE $\partial_t u = -Lu$.

## Running guide

To set up environment first run
```
git clone https://github.com/lclaeys/eigenfunction-solver
cd ./eigenfunction-solver
chmod u+x ./prep.sh
./prep.sh
```