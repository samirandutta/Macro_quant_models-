# Use Pkg.add(.) lines if packages are not already installed
using Pkg
using Parameters 
using Printf
using Distributions
using Roots
using Plots
using LaTeXStrings
using StatsBase  # For the histogram function
using StatsPlots


#========================================================================================================================================================================================================================================================================#
# Parameters + Discretization
#========================================================================================================================================================================================================================================================================#

#Tauchen Function to discretize continuous process 
function tauchen(N, ρ, σ; μ = 0.0, m = 3.0)
	s1    = μ/(1-ρ) - m*sqrt(σ^2/(1-ρ^2))
   	sN    = μ/(1-ρ) + m*sqrt(σ^2/(1-ρ^2))
    s = collect(range(s1, sN, length = N))
    step    = (s[N]-s[1])/(N-1)  #evenly spaced grid
    P      = fill(0.0, N, N)

    for i = 1:ceil(Int, N/2)
    	P[i, 1] = cdf.(Normal(), (s[1] - μ - ρ*s[i] + step/2.0)/σ)
        P[i, N]  = 1 - cdf.(Normal(), (s[N] - μ - ρ*s[i]  - step/2.0)/σ)
        for j = 2:N-1
        	P[i,j]  = cdf.(Normal(), (s[j] - μ - ρ*s[i]  + step/2.0)/σ) -
                            cdf.(Normal(), (s[j] - μ - ρ*s[i] - step/2.0)/σ)
        end
        P[floor(Int, (N-1)/2+2):end, :]=P[ceil(Int ,(N-1)/2):-1:1, end:-1:1]
	end

    ps = sum(P, dims = 2)
    P = P./ps

    return s, P
end


#Setting parameters 
#Note: log AR(1) process for productivity: log(z_t) = (1-ρ) μ + ρ log(z_t-1) + ϵ_t ; where ϵ_t ≈ N(0,σ^2) 
#Production function: y = zn^α

function set_par(
    β = 0.8,                           # Firm's discount factor 
    A = 0.01,                          # Labor Disutility 
    ρ = 0.9,                           # Persistence of AR(1)
    μ = 1,                             # Drift term of AR(1)
    σ = 0.2,                           # Standard Deviation of i.i.d shock (not AR(1))
    α = 2/3,                           # Labor Share
    c_e = 40,                          # Entry cost 
    c_f = 20,                          # Per-period fixed cost 
    τ = 0.2,	                       # Labour Adjustment costs
    D_bar = 100.0,                     # Invariant demand parameter                  
    n_z = 101,                         # Number of grid points for productivity z
    l_min = 0,                         # Min number of workers, note that firms can exit by reducing their labor force to zero! 
    l_max = 100,                       # Max number of workers
    #n_l = l_max,
    w = 1.0,                           # Taking wage as numeraire
    p_adjust = 0.8)                     

    #Discretization 
    μ = (1-ρ)*μ                        # Note: Here, due to the normalisation by multiplication of (1-ρ) in the drift of AR(1), we know that μ = μ_z = mean of AR(1), since μ_z = μ/(1-ρ)
    z_grid, F_trans = tauchen(n_z, ρ, σ; μ = μ, m = 4.0)     
    inv_z = F_trans^1000              # invariant distribution
	inv_z = inv_z[1,:]	

    #ENTRANTS DISTRIBUTION: Assume they draw from the invariant distribution.
	G_z = inv_z
    
    #Productivity State Space
    z_grid = @. exp(z_grid) 

    #Labour State Space 
    # l_grid = collect(range(l_min, l_max, length = n_l))
    # l_grid = round.(Int, l_grid) |> unique                #To ensure integer values for labour 
    l_grid = [collect(0:1:500); collect(550:50:1000); 
			 collect(1100:100:5000);  collect(5500:500:10000)]	
    l_grid = [collect(0:1:20); collect(22:2:100); collect(105:5:500); collect(550:50:1000); 
			 collect(1100:100:5000);  collect(5500:500:10000)]	
    n_l = length(l_grid)

    #Labor adjustment grid: (100 * 100)
    adjust_mat = fill(0.0, n_l, n_l)
    # for il_prev = 1:n_l, il = 1:n_l
    #     adjust_mat[il_prev, il] = τ*max(0, l_grid[il_prev] - l_grid[il])
    # end

    #IDA specification (Strict)
    # for il_prev = 1:n_l, il = 1:n_l
    #    if l_grid[il_prev] >= 100
    #        adjust_mat[il_prev, il] = τ * max(0, l_grid[il_prev] - l_grid[il])
    #    else
    #        adjust_mat[il_prev, il] = 0.0
    #    end
    # end

    #IDA specification (Fuzzy)
    for il_prev = 1:n_l, il = 1:n_l
        if l_grid[il_prev] >= 100
            if rand() < p_adjust  # Apply the adjustment cost with probability p_adjust
                adjust_mat[il_prev, il] = τ * max(0, l_grid[il_prev] - l_grid[il])
            else
            adjust_mat[il_prev, il] = 0.0
            end
        end
    end

    #Production given the z and l grids: (100 * 101)
    Y_mat = fill(0.0, n_l, n_z)
    for jn = 1:n_l, jz = 1:n_z
        Y_mat[jn, jz] = z_grid[jz]*l_grid[jn]^α 
    end

    return (β = β, A = A, α = α, c_e = c_e, c_f = c_f, τ = τ, D_bar = D_bar,  n_z = n_z, n_l = n_l, w = w, 
    G_z = G_z, z_grid = z_grid, l_grid = l_grid, adjust_mat = adjust_mat, Y_mat = Y_mat,  F_trans = F_trans)

end

param = set_par();
@unpack z_grid, l_grid, adjust_mat, Y_mat, F_trans, G_z, α = param               #to check stuff

#scrap_val = - repeat(adjust_mat[:, 1], 1, 101)
# Initializing VF
#V = fill(1, 100, 101)
#EV = V*F_trans'
#CV = 0.8*max.(EV, scrap_val)
#profit_mat = fill(0.0, 100, 101, 100)

#======================================================================================================================================================================================================================================================#
# Solving the Bellman equation (VFI)
#======================================================================================================================================================================================================================================================#

function solve_bellman(p_guess, param; print_it = false)
    @unpack β, c_f, n_z, n_l, z_grid, l_grid, F_trans, adjust_mat, Y_mat = param

    # Iteration prelims 
    tol = 10.0^-5
    max_iter = 1000

    # Aux Matrices
    scrap_val = - repeat(adjust_mat[:, 1], 1, n_z) #Dim: n_l * n_z #This takes the first column of adjust_mat and repeats it for n_z columnsm, the adjust_mat[:,1] column contains all possible cases when adjustment is done to zero workers
    profit_mat = fill(0.0, n_l, n_z, n_l)          #Array of dim: n_l (previous)*n_z, for each possible n_l choice today

    #Static Profit Function
    for il_prev = 1:n_l, il = 1:n_l, iz = 1:n_z
        profit_mat[il_prev, iz, il] = p_guess*Y_mat[il, iz] - l_grid[il] - p_guess*c_f - adjust_mat[il_prev, il] #dim: n_l * n_z for all choices of n_l today 
    end
    
    function vfi_solve(V)
        #Note: V has to be initialised as a n_l * n_z matrix 
        EV = V*F_trans'                     #dim: (n_l * n_z) x (n_z * n_z) = n_l * n_z
        CV = β*max.(EV, scrap_val)          #dim: n_l * n_z
        Vnext = zeros(n_l, n_z)

        #For all labor choices today (il) for each combination of (il_prev, iz)
        for iz = 1:n_z, il_prev = 1:n_l
            Vnext[il_prev, iz] = maximum(profit_mat[il_prev, iz, :] + CV[:, iz])
        end
        return Vnext
    end

    # Initializing VF
    V = fill(0.0, n_l, n_z)
    V_guess = copy(V)

    #Initialize iteration
    iter = 0 

    # Value function iteration
    while iter <= max_iter
        iter += 1
        if print_it; 
            println("VF Iteration #$iter")
        end

        # Update guess
        V_guess .= V

        V = vfi_solve(V_guess)

        # Check for convergence 
        sup = maximum(abs.(V-V_guess))

        if sup < tol
            println("Value function converged! Max. different = $sup")
            break
        end

        if iter == max_iter
            println("Max iterations achieved, VF did not converge! Max. difference = $sup")
        end

    end

    #Recovering policy functions
    EV = V*F_trans'                           #dim: n_l * n_z
    CV = β*max.(EV, scrap_val)                #dim: n_l * n_z

    #Exit policy
    χ = zeros(n_l, n_z)
    χ[scrap_val .>= EV] .= 1.0;                #Those with a negative expected continuation value => exit (χ=1)
    χ[scrap_val .< EV] .= 0.0;                 #Those with a positive expected continuation value => stay (χ=0)

    #Initialise 
    npi = fill(0, n_l, n_z)                    #Note the use of Int type, since we want an integer index for use later in profit_mat[;,;, index]
    profit_opt = fill(0.0, n_l, n_z)           #Optimal profits 
    T = fill(0.0, n_l, n_z)                    #Tax Revenue   

    for il_prev = 1:n_l, iz = 1:n_z 
        npi[il_prev, iz] = argmax(profit_mat[il_prev, iz, :] .+ CV[:, iz])      #Finds the index with the optimal labor choice today, for all states (il_prev, iz)
        profit_opt[il_prev, iz] = profit_mat[il_prev, iz, npi[il_prev, iz]]     #Plugs the above index to find the optimal profits for each state (il_prev, iz) 
        T[il_prev, iz] = adjust_mat[il_prev, npi[il_prev, iz]]*(1-χ[il_prev, iz]) +  #Tax revenue from those who stay and adjust 
                         χ[il_prev, iz]*adjust_mat[il_prev, 1]                       #Tax reveneu from those who exit (i.e., drop labor to 0)
    end

    emp = l_grid[npi]                                       #dim: n_l * n_z, takes the optimal index from npi for given states (emp_prev, prod) and gives the value of emp from l_grid

    return (V = V, npi = npi, χ = χ, emp = emp, profit_opt = profit_opt, T = T, EV = EV, scrap_val = scrap_val)

end


#test_vfi = solve_bellman(1.0, param; print_it = true)


#======================================================================================================================================================================================================================================================#
# Solving for the equilibrium price (Bisection Method)
#======================================================================================================================================================================================================================================================#

function find_eqprice(param)

    @unpack β, c_e, G_z, w = param

    p_min, p_max = 1, 5       #some random guess
    max_iter_p = 500
    iter_p = 0
    p_star = 0.0              # Initialize price 

    while iter_p <= max_iter_p
        iter_p += 1 
        p_guess = (p_min + p_max)/2

        # Call the VFI function
        V, npi, χ, emp, profit_opt, T = solve_bellman(p_guess, param; print_it = true)
        
        # Value of entry 
        V_e = β*sum(V[1,:].*G_z) - c_e  #V[1,:] takes cases of 0 labor with all possible productivities * weighted by G_z and then summed up!
        
        println("Bisection Iteration #", iter_p, "; V_e = ", V_e)

        if abs(V_e) < 0.001
            p_star = p_guess
            println("Equilibrium price found! Equilibrium Price = ", p_guess, ", now iterating on Bellman with this price:")  
            break
        elseif V_e > 0 
            p_max = p_guess
            println("Excess entry, setting p_guess = ", (p_min + p_max)/2, "; p_max = ", p_max, " , p_min = ", p_min)
        elseif V_e < 0
            p_min = p_guess
            println("No entry, setting p_guess = ", (p_min + p_max)/2, "; p_max = ", p_max, " , p_min = ", p_min)
        end


    end

    V, npi, χ, emp, profit_opt, T, EV, scrap_val = solve_bellman(p_star, param; print_it = false)

    return (p_star = p_star, V = V, npi = npi, χ = χ, emp = emp, profit_opt = profit_opt, T = T, EV = EV, scrap_val = scrap_val)
    
end

@time sol_ptest = find_eqprice(param) 
#@unpack χ, npi, emp = sol_ptest

#inv_dist = fill(0.0, 100, 101)
#@unpack G_z = param
#inv_dist[1, :] = G_z

#==================================================================================================================================================================================================================================================================#
# Solving for Invariant Distribution
#==================================================================================================================================================================================================================================================================#

function invariant_dist(param, aux)
    @unpack G_z, F_trans, n_l, n_z = param 
    @unpack χ, npi = aux

    #Initialize 
    inv_dist = fill(0.0, n_l, n_z)
    inv_dist[1, :] = G_z                        #guess
    dist_next = fill(0.0, n_l, n_z)

    function dist_lawofmotion(dist_next, inv_dist)
        for il_prev = 1:n_l, iz = 1:n_z
            dist_next[npi[il_prev, iz], :] = dist_next[npi[il_prev, iz], :] + inv_dist[il_prev, iz]*F_trans[iz, :]*(1-χ[il_prev, iz])
        end
        dist_next[1, :] = dist_next[1, :] + G_z #Adding entrants in the first state of empl i.e., 0
        return dist_next                        
    end

    #Iteration preliminaries
    tol = 10.0^-6
    max_iter_d = 500
    iter_d = 0

    #Iteration
    while iter_d <= max_iter_d
        iter_d += 1 
        println("Invariant Distribution Iteration #$iter_d")

        dist_next .= 0.0                              #Resetting after each iteration

        dist_next = dist_lawofmotion(dist_next, inv_dist)

        sup = maximum(abs.(dist_next - inv_dist))  # check tolerance

        inv_dist .= dist_next                      # Updates inv_dist to reflect iterations 

        if sup < tol
            println("Invariant distribution found!")
            break
        end
        
        if iter_d == max_iter_d
            println("Max iterations done, invariant distribution not found")
        end

    end

    return inv_dist

end

#mu_test = invariant_dist(param, sol_ptest)

#====================================================================================================================================================================================================================================================================================#
# Solving for Equilibrium Mass 
#=====================================================================================================================================================================================================================================================================================#

function solve_M(param, aux)
    @unpack A, w, G_z, l_grid, z_grid, n_l, n_z, α, c_f, c_e = param 
    @unpack p_star, emp, T, profit_opt, χ = aux
    
    # Solve for steady-state distribution
    μ = invariant_dist(param, aux)

    # Solve for M
    Y_opt = emp.^α.*z_grid'                      #dim: (n_l * n_z) .x (1 * n_z) = (n_l * n_z)
    agg_ss_Mequal1 = sum((Y_opt .- c_f).*μ)      #aggregate ss with M = 1 #Note: sum() sums all elements in the matrix, resulting in a scalar
    agg_demand = 1/(A*p_star)
    M = agg_demand/agg_ss_Mequal1

    if M <= 0;
        println("No entry equilibrium")
    else 
        println("M>0, we have an equilibrium with positive entry!")
    end 

    #True invariant distribution with M
    μ = M.*μ

    #Aggregation with M 
    Y = agg_ss_Mequal1*M
    N = sum(emp.*μ) + c_e*M
    R = sum(T.*μ)
    Π = sum(profit_opt.*μ) - c_e*M
    Π_alt = p_star*Y - w*N - R

    if abs(Π - Π_alt) < 1
        println("Profits are consistent with aggregation!")
    else
        println("Profits are not consistent with aggregation..")
    end

    # Exit productivity 
    z_cut = zeros(n_l)
    for i = 1:n_l
        cut_index = findfirst(χ[i,:] .== 0)    #finds the first productivity level at which firms do not exit
        z_cut[i] = z_grid[cut_index]           #productivity exit threshold for all labor states 
    end

    return (M = M, μ = μ, Y = Y, N = N, R = R, Π = Π,  Π_alt = Π_alt, z_cut = z_cut)

end

@time sol_dist = solve_M(param, sol_ptest)
#@unpack z_cut = sol_dist

#================================================================================================================================================================================================================================================================#
# Model Statistics
#================================================================================================================================================================================================================================================================#

function ModelStats(param, sol_ptest, sol_dist)
    @unpack n_l, n_z, z_grid, l_grid, α = param
    @unpack p_star, χ, emp, npi = sol_ptest
    @unpack M, μ = sol_dist

    #productivity distribution
    pdf_z = sum(μ, dims = 1)[:]./sum(μ)
    #pdf_dist2 = sum(μ, dims = 2)[:]./sum(μ)
    #cdf_z = cumsum(pdf_z)

    #employment distribution
    pdf_n = sum(emp.*μ, dims = 2)[:]./sum(emp.*μ)
    emp_weighted = emp.*μ
    #pdf_emp2 = sum(emp.*μ, dims = 2)[:]./sum(emp.*μ)
    #cdf_n = cumsum(pdf_n)

    return (pdf_z = pdf_z, pdf_n = pdf_n, emp = emp, μ = μ, z_grid = z_grid, l_grid = l_grid)

end

model_stats = ModelStats(param, sol_ptest, sol_dist)
@unpack pdf_z, pdf_n, emp, μ, z_grid, l_grid = model_stats

#================================================================================================================================================================================================================================================================#
# Plots
#================================================================================================================================================================================================================================================================#

# 1. Employment distribution
#----------------------------------------------------------------------------------------------------------------------------------------------
# 1.1.
plot(l_grid, pdf_n, xlims = (0,500))

# Plot the kernel density estimate of the data
density(pdf_n, l_grid, label="Smoothed Employment Density", xlabel="Labor (l_grid)", ylabel="Density", lw=2, xlims=(0, 500), bw = 10)

# 1.2
# Flatten the emp matrix into a 1D vector
emp_flat = vec(emp)

# Specify the desired bin width
binwidth = 10

# Calculate the number of bins based on the range of emp_flat and the binwidth
num_bins = ceil(Int, (maximum(emp_flat) - minimum(emp_flat)) / binwidth)

# Plot the histogram with the calculated number of bins and limit x-axis to 500
histogram(emp_flat, bins=num_bins, label="Employment Density", xlabel="Employment", ylabel="Density", lw=2, xlims=(0, 500))


# 2. Value function
#----------------------------------------------------------------------------------------------------------------------------------------------

begin 
    @unpack z_grid, l_grid, n_l, n_z = set_par();
    @unpack V, EV, scrap_val = sol_ptest;
    @unpack z_cut = sol_dist;

    # Extracting values for the last labor state
    z_cutend = z_cut[end];
    scrap_valend = scrap_val[end, :]
    EV_end = EV[end, :]

    plot(z_grid, V[end, :], label = "V", xlabel = L"productivity $z$", ylabel = L"value function $V$", lw =2)
    hline!([0], label = "", linestyle=:dash, color=:black)
    plot!(z_grid, scrap_valend, label = "scrap value")
    plot!(z_grid, EV_end, label = "expected value")
    vline!([z_cutend], label =  "exit threshold", lw = 1)
end


# 3. Inaction Region 
#----------------------------------------------------------------------------------------------------------------------------------------------

begin
    @unpack z_grid, l_grid, n_l, n_z = set_par();
    @unpack npi = sol_ptest 
    
    emp_index = repeat(collect(1:1:n_l), 1, n_z)      #dim: (n_l x n_z)
    
    # example:     
    # 1  1  1     .... n_z
    # 2  2  2     .... n_z
    # 3  3  3     .... n_z
    # 4  4  4     .... n_z
    # .  .  .     .... n_z
    # .  .  .     .... n_z
    # n_l n_l n_l .... n_z

    inaction = 1.0*(npi .== emp_index)               #dim: (n_l x n_z)

    inaction_value = zeros(n_z, 2)                   #dim: (n_z x 2)

	for iz = 1:n_z
		indx = findfirst(inaction[:,iz] .== 1)
		inaction_value[iz, 2] = l_grid[indx]
		indx = findlast(inaction[:,iz] .== 1)
		inaction_value[iz, 1] = l_grid[indx]
	end
	inaction_value

    plot(z_grid, inaction_value, label=["\$n_H(z)\$" "\$n_L(z)\$"] , lw = 2, 
        color = [ :blue :lightblue], linestyle = :dash,  legend=:bottomright)
    xlims!(0,12) 
	ylims!(0,500)
	xlabel!("\$z\$")
	ylabel!("\$n\$")
	title!("Inaction Region")
    #indx_f = findfirst(inaction[:,24] .== 1)
    #indx_l = findlast(inaction[:,24] .== 1)



end

using DataFrames

inaction_df = DataFrame(inaction_value, :auto)
show(inaction_df, allrows = true)

z_grid_df = DataFrame(z_grid = z_grid)
show(z_grid_df, allrows = true)



