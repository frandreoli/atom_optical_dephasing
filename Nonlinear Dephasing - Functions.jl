#############################################################################################################
################## FUNCTIONS TO RANDOMLY SAMPLE POINTS ######################################################
#############################################################################################################
#
#
#Function to uniformly sample in a cylinder with main axis oriented along z
@everywhere function uniform_sampling_cylinder!(r_atoms,n_atoms, rLength, zLength)
    length(r_atoms[:,1])!=n_atoms ? error("Incompatible number of atoms in positions sampling.") : nothing
    length(r_atoms[1,:])!=3       ? error("Wrong vector space dimension in positions sampling.") : nothing
    phasesRandom=2.0*pi*rand(n_atoms)
    radiusRandom=sqrt.(rand(n_atoms))
    #
    #See: http://mathworld.wolfram.com/DiskPointPicking.html
    xPoints=rLength*radiusRandom.*cos.(phasesRandom)
    yPoints=rLength*radiusRandom.*sin.(phasesRandom)
    zPoints=rand(n_atoms).*(zLength).-(zLength/2)
    for i in 1:n_atoms
        r_atoms[i,:]=[xPoints[i];yPoints[i];zPoints[i]]
    end
  end
  #
  #Function to uniformly sample in a 3D ball/sphere
  @everywhere function uniform_sampling_sphere!(r_atoms,n_atoms, r_sphere, type_sampling="BALL",hemisphere="FULL_SPHERE")
    length(r_atoms[:,1])!=n_atoms ? error("Incompatible number of atoms in positions sampling.") : nothing
    length(r_atoms[1,:])!=3       ? error("Wrong vector space dimension in positions sampling.") : nothing
    #
    #See: 
    #http://mathworld.wolfram.com/SpherePointPicking.html 
    #http://mathworld.wolfram.com/DiskPointPicking.html 
    #http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    #
    thetaRandom=(2.0*pi).*rand(n_atoms)
    uRandom=2.0.*rand(n_atoms).-1.0
    #
    #Deciding if spanning the full solid angle or half (in the forward/backward direction)
    if hemisphere!="FULL_SPHERE" && hemisphere!="FORWARD_HEMISPHERE" && hemisphere!="BACKWARD_HEMISPHERE"
      error("Invalid definition of hemisphere in uniform_sampling_sphere!()")
    end
    hemisphere=="FORWARD_HEMISPHERE"  ? uRandom=abs.(uRandom)  : nothing
    hemisphere=="BACKWARD_HEMISPHERE" ? uRandom=-abs.(uRandom) : nothing
    #
    #Deciding if sampling a ball or a sphere
    if type_sampling!="BALL" && type_sampling!="SPHERE" 
      error("Invalid definition of type_sampling in uniform_sampling_sphere!()")
    end
    if type_sampling=="BALL"
      rRandom=(rand(n_atoms)).^(1/3)
    else
      rRandom=[1.0 for i in 1:n_atoms]
    end
    #
    #Calculating the points
    xPoints=r_sphere.*rRandom.*sqrt.(1.0.-(uRandom).^2).*cos.(thetaRandom)
    yPoints=r_sphere.*rRandom.*sqrt.(1.0.-(uRandom).^2).*sin.(thetaRandom)
    zPoints=r_sphere.*rRandom.*uRandom
    #
    #Storing them into an array
    for i in 1:n_atoms
        r_atoms[i,:]=[xPoints[i];yPoints[i];zPoints[i]]
    end
  end


#############################################################################################################
################## DISORDERED POSITIONS  ####################################################################
#############################################################################################################
#
#
#Function to define the atomic positions by uniformly sampling inside a sphere
@everywhere function dis_sphere_creation(r_sphere,atomic_density)
  sphere_volume = (4/3)*pi*r_sphere^3
  n_atoms = Int(round(sphere_volume*atomic_density))
  r_atoms = Array{Float64}(undef, n_atoms, 3)
  uniform_sampling_sphere!(r_atoms,n_atoms, r_sphere, "BALL","FULL_SPHERE")
  return (r_atoms,n_atoms)
end
#
#
#Function to define the atomic positions by uniformly sampling inside a cylinder with main axis in the z direction
@everywhere function dis_cyl_creation(r_disk,z_length,atomic_density)
  cylinder_volume = pi*r_disk^2*z_length
  n_atoms = Int(round(cylinder_volume*atomic_density))
  r_atoms = Array{Float64}(undef, n_atoms, 3)
  uniform_sampling_cylinder!(r_atoms,n_atoms, rLength, zLength)
  return (r_atoms,n_atoms)
end
#
#
#Function to define the atomic positions by uniformly sampling inside a cylinder with main axis in the z direction
@everywhere function dis_cuboid_creation(x_dim,y_dim,z_dim,atomic_density)
  cuboid_volume = x_dim*y_dim*z_dim
  n_atoms = Int(round(cuboid_volume*atomic_density))
  r_atoms = Array{Float64}(undef, n_atoms, 3)
  for ii in 1:3
    r_atoms[:,ii].=(rand(Float64,n_atoms).*2.0 .- 1.0)*([x_dim ; y_dim ; z_dim][ii]/2)
  end
  return (r_atoms,n_atoms)
end

#############################################################################################################
################## SAVING FUNCTIONS #########################################################################
#############################################################################################################
#
#
#FUNCTIONS TO SAVE DATA FILES IN HDF5 FORMAT:
#
#Function to save any variable. 
#"cw" -> Do not overwrite existing data
#"w"  -> Overwrite existing data 
@everywhere function h5write_basic(file_name,data, name_variable="" ; open_option="cw")
  file_h5=h5open(file_name*".h5", open_option)
  haskey(file_h5, name_variable) ? delete_object(file_h5, name_variable) : nothing
  file_h5[name_variable]=data
  close(file_h5)
end
#
#Function to save an array of complex number.
#In the HDF5 file the variable "_re" ("_im") contain an array with the the real (imaginary) parts
@everywhere function h5write_complex(file_name,data, name_variable="" ; open_option="cw")
  length(name_variable)>0 ? add_name=name_variable*"_" : add_name = ""
  file_h5=h5open(file_name*".h5", open_option)
  haskey(file_h5, name_variable*"re") ? delete_object(file_h5, name_variable*"re") : nothing
  haskey(file_h5, name_variable*"im") ? delete_object(file_h5, name_variable*"im") : nothing
  file_h5[name_variable*"re"]=real.(data)
  file_h5[name_variable*"im"]=imag.(data)
  close(file_h5)
end
#
#Function to save multiples new variables into a file. 
#If the file already exists it preserves the data, otherwise it creates the file
#If the variable already exist in the file, it overwrites it
#data_array is an array of tuples whose first element is the name of the variable
#while the second is the variable to save. 
#The option open_option="cw" only overwrites the chosen variable in the .h5 file,
#while setting open_option="w" first deletes all elements stored in the file.
@everywhere function h5write_multiple(file_name,data_array... ; open_option="cw")
  file_h5=h5open(file_name*".h5", open_option)
  for index in 1:length(data_array)
    name_variable = data_array[index][1]
    variable_data = data_array[index][2]
    haskey(file_h5, name_variable) ? delete_object(file_h5, name_variable) : nothing
    file_h5[name_variable]=variable_data
  end
  close(file_h5)
end
#
#
#Functions to add new elements to an array already stored
@everywhere function h5write_append(file_name,data, name_variable="")
  open_option="cw"
  file_h5=h5open(file_name*".h5", open_option)
  if haskey(file_h5, name_variable)
    old_data = read(file_h5[name_variable])
    delete_object(file_h5, name_variable)
    try
      file_h5[name_variable]=vcat(old_data, data)
    catch error_writing
      close(file_h5)
      error(error_writing)
    end
  else
    try
      file_h5[name_variable]=data
    catch error_writing
      close(file_h5)
      error(error_writing)
    end
  end
  close(file_h5)
end
#
@everywhere function h5write_complex_append(file_name,data, name_variable="")
  length(name_variable)>0 ? add_name="_" : add_name = ""
  for domain_f in ((real,"re") , (imag, "im"))
    h5write_append(file_name,(domain_f[1]).(data), name_variable*add_name*domain_f[2]) 
  end
end
#
#
#
#
#
#
#