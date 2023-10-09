# Introduction
Simulation of the time dynamics of an atomic spin-wave, either matched $|\mathbf k|=k_0$ or mismatched $|\mathbf k|\neq k_0$. A dephasing phenomenon is observed, due to inter-atomic, optical interactions 
The time dynamics is computed via a first-order cumulant expansion (mean field regime), which accounts for part of the atomic (quantum) nonlinearity. In the limit of low atomic excitation, this is equivalent to the linear (classical) regime of coupled dipoles. The time evolution under this latter condition can be analogously  

# Code features
The time dynamics exploits the package [OrdinaryDiffEq](https://docs.sciml.ai/OrdinaryDiffEq/stable/), allowing to choose between various solvers. 
The simulation can be performed given different atomic densities and number of initial excitations. These different simulations can be performed either sequentially or in parallel, via the `@distributed` macro (see [Multi-processing and Distributed Computing](https://docs.julialang.org/en/v1/manual/distributed-computing/) in Julia). The user can choose between the two cases via the `option_distributed_main` option.
