#############################################################################################################
################## CORE FUNCTIONS ###########################################################################
#############################################################################################################
#
#@everywhere means that the function is defined over all the cores, for parallel computation
#Main
@everywhere function main_loop!( i_rep,j_rep,l_rep,final_path_name, r_sphere, atomic_density, n_atoms_list, dephasing_value, pop_value, overlap_value, t_min, t_max, t_steps, gamma_prime, solver,dipoles_polarization, k_spin_wave, inhom_broad_std, n_excit)
    time_core = time()
	#Creates the random positions
    (r_atoms, n_atoms) = dis_sphere_creation(r_sphere,atomic_density)
	#
	#Saves an array with the number of atoms
    if i_rep==1
        n_atoms_list[j_rep] = n_atoms
        h5write_multiple(final_path_name*"/results", ("n_atoms_list" ,n_atoms_list) )
    end
	#
	#Core simulation
    (dephasing_value[l_rep,j_rep,i_rep,:],pop_value[l_rep,j_rep,i_rep,:],overlap_value[l_rep,j_rep,i_rep,:]) = core_evolution(r_atoms, n_atoms, t_min, t_max, t_steps, gamma_prime, solver,dipoles_polarization, k_spin_wave, inhom_broad_std,n_excit)
    #
	#For very small systems (N_atoms<50) the calculation is very fast. 
	#If many cores are running at the same time, then they might try to write on the data file at the same time
	#returning an error. The following is an attempt to avoid so, delaying the calculation of 1sec if it is too fast.
	#In any case, for small systems I discourage the use of parallel computing
	if option_distributed_main
		time()-time_core<1 ? sleep(1) : nothing
	end
	#
	#This saves the data files into the file results.h5 
	#The data are stored in the format HDF5
	h5write_multiple(final_path_name*"/results", ("overlap_value" ,overlap_value) )
    h5write_multiple(final_path_name*"/results", ("dephasing_value" ,dephasing_value) )
    h5write_multiple(final_path_name*"/results", ("pop_value" ,pop_value) )
    println("Core evaluation time: ", time()-time_core, ".  Number of atoms: ", n_atoms)
end
#
#
@everywhere function core_evolution(r_atoms, n_atoms, t_min,t_max, t_steps, gamma_prime, solver,dipoles_polarization,k_spin_wave,inhom_broad_std,n_excit)
    # 
	#Initializes the G matrix
    G_matrix_RE  =  Array{TN,2}(undef, n_atoms, n_atoms)
	G_matrix_IM  =  Array{TN,2}(undef, n_atoms, n_atoms)
	G_matrix_RE[:,:].=TN(0.0)
	G_matrix_IM[:,:].=TN(0.0)
    CD_initialize!(G_matrix_RE,G_matrix_IM, r_atoms, dipoles_polarization,gamma_prime,inhom_broad_std) 
    #
	#Initializes the spin wave at t=0
	#Here sin(theta/2)=sqrt(n_excit/n_atoms) and cos(theta/2)=sqrt(1-(n_excit/n_atoms))
    spin_wave0 = sqrt(1-(n_excit/n_atoms)).*exp.(((x,y,z)->(-1.0im*(k_spin_wave[1]*x + k_spin_wave[2]*y + k_spin_wave[3]*z))).(r_atoms[:,1], r_atoms[:,2], r_atoms[:,3])) .*sqrt(n_excit/n_atoms) 
    alpha0 = real.(spin_wave0)
    beta0 =  imag.(spin_wave0)
    xi0 = [(2*n_excit/n_atoms)-1 for i in 1:n_atoms]
    #
	#Creates an array u0 where all the initial conditions are stored.
	#From 1 to n_atoms it contains alpha0. From n_atoms+1 to 2*n_atoms it contains beta0.
	#From 2*n_atoms+1 to 3*n_atoms it stores xi0
	#The time evolution will evolve such long array containing all the dyamical variables
    u0 = Array{Float64,1}(undef, 3*n_atoms)
    u0[1:n_atoms]               = alpha0[:]
    u0[n_atoms+1:2*n_atoms]     = beta0[:]
    u0[2*n_atoms + 1:3*n_atoms] = xi0[:]
    #
	#It seems that the computation is fast enough even if the Jacobian is not explicitly defined
	#So I'd keep option_explicit_jac==false, for simplicity and to avoid errors coming from typos
	if option_explicit_jac
		ode_func = ODEFunction(time_step_func!;jac = jacobian_func!)
	else
		ode_func = ODEFunction(time_step_func!)
	end
	#Definition of the differential equation for time evolution
    ode = ODEProblem(ode_func,u0,(t_min,t_max),(G_matrix_RE,G_matrix_IM, n_atoms))
	#Main simulation
	evolved_solutions = solve(ode, solver , saveat=range(t_min, stop=t_max, length=t_steps) )
	#
	#Initializes the arrays where the results will be stored
	#Tr[rho0 rho]
    overlap_saves = Array{Float64}(undef, t_steps)
	#O(t)
	dephasing_saves = Array{Float64}(undef, t_steps)
	#P(t)/N
	population_saves = Array{Float64}(undef, t_steps)
	#
	#It computes the final observables out of the evolved coefficients
    for i in 1:t_steps
        (overlap_saves[i],dephasing_saves[i]) = target_operator((evolved_solutions.u)[i], alpha0, beta0, xi0,n_atoms)
		#
		if option_non_linearity
			populations_temp = (x->(1+x)/2).(((evolved_solutions.u)[i])[2*n_atoms + 1:3*n_atoms])
			positive_pops = (x->x>0).(populations_temp)
			#Sometimes this warning pops up, but as long as the "negative population" is very very small then this is 
			#non a problem, as it is probably a sign of floating-point fluctuactions over zero
			!prod(positive_pops) ? println("Negative population error at i=", i, ". The value of populations is: ", populations_temp[(x->!x).(positive_pops)]) : nothing
			population_saves[i] = sum(populations_temp)/n_atoms
		else
			populations_temp = (abs2.(((evolved_solutions.u)[i])[1:n_atoms] .+ (1.0im.*((evolved_solutions.u)[i])[n_atoms+1:2*n_atoms])))
			population_saves[i] = sum(populations_temp)/n_atoms
		end
    end
	#
	#
    return (dephasing_saves,population_saves,overlap_saves)
end
#
#This function converts the values inside u(t) into the wanted observables
@everywhere function target_operator(u, alpha0, beta0, xi0, N)
	#
    alpha = u[1:N]
    beta = u[N+1:2*N]
    xi = u[2*N + 1:3*N]
	#
	overlap = prod(0.5 .+ (0.5.* xi0.*xi) .+ 2.0.*( alpha0.*alpha .+ beta0.*beta ) ) 
	#
	dephasing = abs2.(sum( (alpha0.-1.0im.*beta0).* (alpha.+1.0im.*beta) ))
	#
	(overlap, dephasing)
end
#
#Time-step of the dynamical evolution
@everywhere function time_step_func!(du,u,p,t)
	#p contains the parameters of the simulation
    reG,imG,N = p
    #
    alpha = u[1:N]
    beta = u[N+1:2*N]
    xi = u[2*N+1:3*N]
    #
    F_vec = reG *beta  .+ imG * alpha 
    G_vec = imG *beta  .- reG * alpha 
    #
	#Differential steps for the variables alpha (1:n_atoms), beta (n_atoms+1:2*n_atoms) and xi (2*n_atoms+1:3*n_atoms)
    du[1:N]     .= (-0.5).*alpha .+ (F_vec.*  xi)   
    du[N+1:2*N] .= (-0.5).*beta  .+ (G_vec.*  xi)  
	#
	#To use the linear equations (i.e. option_non_linearity==false), then one forces dxi=0. 
	#This is added only to directly simulate Stefano's equations
	if option_non_linearity
		du[2*N+1:3*N] .= (-1.0) .- xi .- 4.0.*(F_vec.*alpha .+ G_vec.*beta)
	else
    	du[2*N+1:3*N] .= 0.0 
	end
    #
end
#
#This function is used to explicitly feed the time evolution with an analytical Jacobian
#when option_explicit_jac==true. I suggest to keep option_explicit_jac==false and not to use this function.
@everywhere function jacobian_func!(J,u,p,t)
	reG,imG,N = p
	#
	alpha = u[1:N]
    beta = u[N+1:2*N]
    xi = u[2*N+1:3*N]
	#
	if option_non_linearity
		F_vec = reG *beta  .+ imG * alpha 
    	G_vec = imG *beta  .- reG * alpha 
	end
	#
	@inbounds for i in 1:N, j in 1:N
		if i==j
			J[i,i]     = -0.5
			J[N+i,N+i] = -0.5
			#
			if option_non_linearity
				J[2*N+i,2*N+i]  = -1.0
				J[i,2*N+i]      = F_vec[i]
				J[N+i,2*N+i]    = G_vec[i]
				J[2*N+i,j]      = -4*F_vec[i]
				J[2*N+i,N+j]    = -4*G_vec[i]
			end
		else
			J[i,j]   = imG[j,i]*xi[i]
			J[i,N+j] = reG[j,i]*xi[i]
			J[N+i,j]   = -reG[j,i]*xi[i]
			J[N+i,N+j] = imG[j,i]*xi[i]
			#
			if option_non_linearity
				J[2*N+i,j] = -4*(alpha[i]*imG[j,i]- beta[i]*reG[j,i])
				J[2*N+i,N+j] = -4*(alpha[i]*reG[j,i]+ beta[i]*imG[j,i])
			end
			#
		end
	end
	#
end
#
#Initialization of the G matrix. 
#If option_distributed_main == false then the code performs the simulations with different atomic positions
#sequentially, otherwise on multiple cores. If these simulations are done sequentially, then it make sense that 
#this specific operation is parallelized on multiple threads. On the other hand, if the different simulations
#are already done in parallel over different cores, then this function (which is inside each simulation) cannot
#run in parallel over different threads, to avoid the saturation of the cluster. 
@everywhere function CD_initialize!(G_matrix_RE, G_matrix_IM, r_vecs, p,gamma_prime,inhom_broad_std)
    na = length(r_vecs[:,1])
	r_vecs_x = r_vecs[:,1].*k0
	r_vecs_y = r_vecs[:,2].*k0
	r_vecs_z = r_vecs[:,3].*k0
	#Total number of steps
	n_steps =Int((na^2 + na)/2)
	#The core part is constructed as a single cycle loop to make the parallelization in threads faster
	#The code is structured not to require the loading of external functions, to boost the parallelization efficiency
	if !option_distributed_main
		Threads.@threads for index in  1:n_steps
			#Constructing two indices from a single index
			i = Int(floor( -(1/2) + sqrt(1/4 + 2*(index - 1 ))   )) + 1
			j = Int(index - (i - 1)*i/2)
			#Inhomogeneous broadening option
			inhom_broad_std>0.0 ? inhom_broad_here=randn()*inhom_broad_std : inhom_broad_here=0.0
			#Constructing the values
			x_i=r_vecs_x[i]
			y_i=r_vecs_y[i]
			x_j=r_vecs_x[j]
			y_j=r_vecs_y[j]
			z = r_vecs_z[i]- r_vecs_z[j]
			#Value of the Greens's function
			if i==j
				value_temp = (0.5im*gamma_prime) + inhom_broad_here
			else
				x = x_i-x_j
				y = y_i-y_j
				r =sqrt((x^2)+(y^2)+(z^2))
				cos_theta_square = ((x*p[1])+(y*p[2])+(p[3]*z))^2
				value_temp =(3*exp(im*r)/(4*r^3)*((r^2 + im*r - 1)*1+(3 - r^2 - 3im*r)*cos_theta_square/r^2))
			end
			#
			G_matrix_RE[i, j] = G_matrix_RE[j, i] = TN(real(value_temp))
			G_matrix_IM[i, j] = G_matrix_IM[j, i] = TN(imag(value_temp))
		end
	else
		for i in 1:na, j in i+1:na
			inhom_broad_std>0.0 ? inhom_broad_here=randn()*inhom_broad_std : inhom_broad_here=0.0
			x_i=r_vecs_x[i]
			y_i=r_vecs_y[i]
			x_j=r_vecs_x[j]
			y_j=r_vecs_y[j]
			z = r_vecs_z[i]- r_vecs_z[j]
			x = x_i-x_j
			y = y_i-y_j
			r =sqrt((x^2)+(y^2)+(z^2))
			cos_theta_square = ((x*p[1])+(y*p[2])+(p[3]*z))^2
			value_temp = (3*exp(im*r)/(4*r^3)*((r^2 + im*r - 1)*1+(3 - r^2 - 3im*r)*cos_theta_square/r^2))
			G_matrix_RE[i, j] = G_matrix_RE[j, i] = TN(real(value_temp))
			G_matrix_IM[i, j] = G_matrix_IM[j, i] = TN(imag(value_temp))
		end
		for i in 1:na
			inhom_broad_std>0.0 ? inhom_broad_here=randn()*inhom_broad_std : inhom_broad_here=0.0
			G_matrix_RE[i,i]=TN(inhom_broad_here)
			G_matrix_IM[i,i]=TN(0.5*gamma_prime)
		end
	end
end
#
#
#
#
#
########################################################################################
#
#solve() options 
#=
tstops: Denotes extra times that the timestepping algorithm must step to. 
This should be used to help the solver deal with discontinuities and singularities, 
since stepping exactly at the time of the discontinuity will improve accuracy. 
If a method cannot change timesteps (fixed timestep multistep methods), 
then tstops will use an interpolation, matching the behavior of saveat. 
If a method cannot change timesteps and also cannot interpolate, 
then tstops must be a multiple of dt or else an error will be thrown. Default is [].
=#
#=
saveat: Denotes specific times to save the solution at, during the solving phase. 
The solver will save at each of the timepoints in this array in the most efficient 
manner available to the solver. If only saveat is given, then the arguments save_everystep 
and dense are false by default. If saveat is given a number, then it will automatically 
expand to tspan[1]:saveat:tspan[2]. For methods where interpolation is not possible, 
saveat may be equivalent to tstops. The default value is [].
=#


 