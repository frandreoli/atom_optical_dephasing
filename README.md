# Introduction
Simulation of the time dynamics of an atomic spin-wave, either matched $|\mathbf k|=k_0$ or mismatched $|\mathbf k|\neq k_0$. An intrinsic dephasing phenomenon is expected, due to inter-atomic, optical interactions between the atoms [[1](Grava2022RenormalizationMedium)]. The time dynamics is computed via a first-order cumulant expansion (mean-field regime), which accounts for part of the atomic (quantum) nonlinearity. This is equivalent to an ansatz of locally separable quantum state, so that the initial spin wave can be written as

$$
\left|\psi (\mathbf k) \right\rangle= \displaystyle\bigotimes\_{j=1}^N\left[ \left(\sqrt{1-\dfrac{M}{N}}\right) \left|g\_j \right\rangle + \left(\sqrt{\dfrac{M}{N}}\right) e^{i\mathbf k\cdot \mathbf{r}\_j }\left|e\_j \right\rangle  \right],
$$
where $N$ is the number of atoms, while $M$ is the average number of excitations (i.e. $M/N$ is the single-atom excitation probability). Here, we also defined $\left|g\_j \right\rangle$ and $\left|e\_j \right\rangle$ as respectively the ground and excited states of the $j$-th atom.

In the limit of low atomic excitations ($M/N\ll 1$), this analysis is equivalent to the linear (classical) regime of coupled dipoles. In case one wants to compare the cumulant expansion described above with the results of the linear approximation, the code allows to compute the time evolution under this latter (linear) regime as well, by changing the option `option_non_linearity` from `true` (nonlinear mean-field) to `false` (linear regime).


# Code features
The time dynamics exploits the package [OrdinaryDiffEq](https://docs.sciml.ai/OrdinaryDiffEq/stable/), allowing to choose between various solvers

```(nothing, Tsit5(), RK4(), lsoda(), Vern8(), QNDF(), QNDF(autodiff=false), Vern9())```

where `nothing` stands for the automatic solver. The Jacobian of the time evolution can be either computed numerically or explicitly fed to the ODE solver.

The simulation can be performed for different atomic densities and number of initial excitations. These different simulations can be performed either sequentially or in parallel, via the `@distributed` macro (see [Multi-processing and Distributed Computing](https://docs.julialang.org/en/v1/manual/distributed-computing/) in Julia). The user can choose between the two possibilities via the `option_distributed_main` option, depending on the hardware resources. 

# References
<a id="Grava2022RenormalizationMedium">[11]</a>
Grava S., He Y., Wu S. and Chang D. E.,
*Renormalization group analysis of near-field induced dephasing of optical spin waves in an atomic medium*,
[New Journal of Physics 24, 013031](https://iopscience.iop.org/article/10.1088/1367-2630/ac465d) (2022)
