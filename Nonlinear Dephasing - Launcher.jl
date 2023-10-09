#############################################################################################################
################## INITIAL OPERATIONS #######################################################################
#############################################################################################################
#
#Designed for Julia 1.8
using Distributed
@everywhere using LinearAlgebra, Dates, HDF5, Random
time_lib = time()
@everywhere using OrdinaryDiffEq, LSODA  #DifferentialEquations #SciMLBase, 
time_lib = time() - time_lib
@everywhere using SharedArrays
#
Random.seed!()
#
if nworkers()==1 addprocs() end
BLAS.set_num_threads(nworkers())
const ZERO_THRESHOLD = 10^(-12)
#For large-scale numerics Float32 is preferable compared to the default Float64
@everywhere const TN = [Float32 ; Float64][1]
#
include("Nonlinear Dephasing - Functions.jl")
include("Nonlinear Dephasing - Core.jl")
#
#Uncomment the following line only if the code seems to be leaking RAM memory or spending too much time in 
#garbage collection. You will force the code to print a line any time the unused memory gets emptied.
#These messages will appear in the form "GC: ..."
#
#GC.enable_logging(true)
#
#
length(ARGS)>=1 ? args_checked=ARGS[:] : args_checked=["TEST"] 



#############################################################################################################
################## OPTIONS ##################################################################################
#############################################################################################################
#
#If the following option is true then the code tries to run each simulation (with different, randomly positioned atoms)
#on a different core, to speed up the computation
@everywhere const option_distributed_main    =  [true ; false][1]
#
#If this option is set to false then the code performs the simpler linear simulations
@everywhere const option_non_linearity       =  [true ; false][1]
#
#This option feeds the time evolution with an explicit Jacobian matrix. It turns out that this doesn't speed up the code
#much, so I'd keep it to NO
@everywhere const option_explicit_jac        =  [true ; false][2]
#



#############################################################################################################
################## FIXED PARAMETERS #########################################################################
#############################################################################################################
#
@everywhere const k0 = 2.0*pi
#Orientation of the atomic dipoles:
@everywhere const dipoles_polarization = [1.0 ; 0.0 ; 0.0]
#In case one wants to add some inelastic losses then set it >0. For normal simulations keep it to 0.0
gamma_prime       =    0.0
#In case one wants to add some inhomogeneous broadening of the resonant frequencies of the atoms. 
#This would give the standard deviation (Gaussian distribution). If =0.0 then no inhomogeneous broadening is added.
inhom_broad_std   =    0.0
#


#############################################################################################################
################## PARAMETERS ###############################################################################
#############################################################################################################
#
#Number of repetitions (each has a randomly chosen set of atomic positions)
n_repetitions     =    30
#List of values of densities that one wants to compute
atomic_density    =    [10]
#Radius of the sphere inside which the atomic positions are randomly chosen
r_sphere          =    2.9
#Initial time
t_min             =    0.0
#Final time
t_max             =    25
#Number of time steps
t_steps           =    250 
#The following decides which solver use. 3, i.e. Runge-Kutta 4th order, seems quite fine
solver_index      =    3
solver            =    (nothing, Tsit5(), RK4(), lsoda(), Vern8(), QNDF(),QNDF(autodiff=false),Vern9())[solver_index] #lsoda(),
#Initial wavevector of the mismatched spinwave
k_spin_wave       =    k0.*[0.0 ; 0.0 ; 6.0]
#Number of excitations that one wants inside the system. The excited-state coefficient is Sin(theta/2)=sqrt(n_excitations/n_atoms)
#If n_excitations = 1 means the single excitation (linear) case. For each value of this list, the system will simulate the whole process.
n_excitations     =    [1 2 4 10 20 50 100 200]
#
!option_non_linearity ? n_excitations=[1] : nothing


################## FILENAME DEFINITION ##############################################################################################################################
#
#Defines a file-name for the simulation
file_name="Dephasing"
file_name*="_r"*string(r_sphere)
if length(atomic_density)>1
    file_name*="_eta"*string(atomic_density[1])*"-"*string(atomic_density[end])
else
    file_name*="_eta"*string(atomic_density[1])
end
file_name*="_tMax"*string(t_max)
file_name*="_tSteps"*string(t_steps)
#
solver_index>1 ? file_name*="_solv"*string(solver_index) : nothing
#
if abs(norm(k_spin_wave./k0)-1.0)<ZERO_THRESHOLD
    file_name*="_MATCHED"
end
if !option_non_linearity
    file_name*="_LINEAR"
end
if option_explicit_jac
    file_name*="_ExpJac"
end
minute_name=string(minute(now()))
length(minute_name)<2 ? minute_name="0"*minute_name : minute_name
file_name*="_d"*string(today())*"_h"*string(hour(now()))*"."*minute_name*args_checked[1]



################## STARTING EVALUATION #################################################################################################################################
#
println("\n\n\n\n\n\n\n\nTime: ",now(),"\nStarting evaluation of ", @__FILE__,"\n\n")
println("Loaded packages in ", time_lib)
println("Output file name: ",file_name,"\n\n")
time_in_0=time()
final_path_name="Data/"*file_name
mkpath(final_path_name)
#
# 
#Creates the arrays where the data are stored
#If one wants to parallelize (i.e. if option_distributed_main==true), then the arrays must be initialized as 
#shared arrays between the cores
h5write_multiple(final_path_name*"/inputs", ("n_repetitions",n_repetitions), ("r_sphere",r_sphere), ("gamma_prime", gamma_prime), ("atomic_density",atomic_density) , ("n_excitations",n_excitations), ("t_max",t_max), ("t_steps",t_max) ; open_option="w")
if !option_distributed_main
    overlap_value = Array{Float64}(undef,length(n_excitations),length(atomic_density),n_repetitions,t_steps)
    dephasing_value = Array{Float64}(undef,length(n_excitations),length(atomic_density),n_repetitions,t_steps)
    pop_value = Array{Float64}(undef,length(n_excitations),length(atomic_density),n_repetitions,t_steps)
    n_atoms_list = Array{Float64}(undef,length(atomic_density))
else
    overlap_value = SharedArray{Float64}((length(n_excitations),length(atomic_density),n_repetitions,t_steps))
    dephasing_value = SharedArray{Float64}((length(n_excitations),length(atomic_density),n_repetitions,t_steps))
    pop_value = SharedArray{Float64}((length(n_excitations),length(atomic_density),n_repetitions,t_steps))
    n_atoms_list = SharedArray{Float64}((length(atomic_density)))
end
overlap_value[:,:,:,:].=0.0
dephasing_value[:,:,:,:].=0.0
pop_value[:,:,:,:].=0.0
n_atoms_list[:].=0.0
time_steps = collect(range(0,t_max,t_steps))
h5write_multiple(final_path_name*"/results", ("time_steps" ,time_steps)  ; open_option="w")
#
#
time_start = time()
#
#Main computation
for j_rep in 1:length(atomic_density)
    println("Starting density ", j_rep,"/",length(atomic_density))
    for l_rep in 1:length(n_excitations)
        println(" - Starting excitation ", l_rep,"/",length(n_excitations))
        if !option_distributed_main
            for i_rep in 1:n_repetitions
                main_loop!( i_rep,j_rep,l_rep,final_path_name,r_sphere, atomic_density[j_rep], n_atoms_list,dephasing_value, pop_value, overlap_value, t_min, t_max, t_steps, gamma_prime, solver,dipoles_polarization, k_spin_wave, inhom_broad_std, n_excitations[l_rep])
            end
        else
            @sync @distributed for i_rep in 1:n_repetitions
                main_loop!( i_rep,j_rep,l_rep,final_path_name,r_sphere, atomic_density[j_rep], n_atoms_list,dephasing_value, pop_value, overlap_value, t_min, t_max, t_steps, gamma_prime, solver,dipoles_polarization, k_spin_wave, inhom_broad_std, n_excitations[l_rep])
            end
        end
    end
end

println("\nTotal evaluation finished in: ", time()-time_start)