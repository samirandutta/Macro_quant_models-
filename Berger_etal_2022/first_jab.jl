#______________________________________________________________________________________________________________________________________________________________________________________________________________
# Labor Market Power, Berger et al. (2022)
# Note: This code runs the baseline model with the original paper's calibrations 
# Author: Samiran Dutta (12.02.25)
#______________________________________________________________________________________________________________________________________________________________________________________________________________

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Packages
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

using Parameters 
using Printf
using Distributions
using Roots
using Plots
using LinearAlgebra
using SparseArrays
using LaTeXStrings
using Statistics
using StatsBase
using Pkg
using Random

# Set seed for reproduction 
Random.seed!(191219981219)  

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Parameters
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

function set_par(;
    𝜑 = 0.50,
    α = 0.9263,
    β = 1/1.04,
    δ = 0.1,
    η = 3.74,
    θ = 0.76,
    J = 10000,
    τ = 10, 
    min_ij = 1,
    max_ij = 50, 
    r = 0.04,
    R = (1/β) - 1 + δ, 
    υ = 0.20,                 #Adjustment rate for shares in equilib solver
    γ = 0.818,
    α_tilde = 0.984,
    Z_tilde = 23570,
    𝜑_tilde = 6.904) 

    return (𝜑 = 𝜑, α = α, β = β, η = η, θ = θ, J = J, τ = τ, min_ij = min_ij, max_ij = max_ij, 
            r = r, R = R, υ = υ, γ = γ, α_tilde = α_tilde, Z_tilde = Z_tilde, 𝜑_tilde = 𝜑_tilde)
    
end

param = set_par();

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Auxilliary Functions 
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# 1. Draw number of firms Mj in each market j = 1,..., J
# ------------------------------------------------------------------------------------------------------

function gen_firms(param)
    @unpack J, min_ij, max_ij = param

    # Initialize
    Mj = zeros(Int, J)
    α_pareto = 2.2

    # Generate firm numbers using Pareto distribution
    pareto_dist = Pareto(min_ij, α_pareto)
    Mj .= ceil.(rand(pareto_dist, J))               # Use ceil to ensure integer values ≥ min_ij
    Mj .= min.(Mj, max_ij)                          # Ensure max cap is applied

    # Randomly pick one market and assign 1 firm
    Mj_1 = rand(1:J)
    Mj[Mj_1] = 1  

    # Update max_ij to be the maximum observed Mj
    Mj_max = maximum(Mj)

    return (Mj = Mj, Mj_max = Mj_max)
end


# 2. Draw firm-level productivites, for each i in j = 1,..,J
# ------------------------------------------------------------------------------------------------------

function gen_productivity(param)
    @unpack J, γ, α, R = param

    # Generate number of firms per market
    firms = gen_firms(param)
    @unpack Mj, Mj_max = firms 

    # Initialise 
    z_ij_raw = zeros(Mj_max, J)
    z_ij = zeros(Mj_max, J)
    xi = 0.5                                                # Standard deviation of lognormal distribution

    # Generate lognormal draws for each i in each j
    lognorm_dist = LogNormal(1, xi)
    for j in 1:J
        z_ij_raw[1:Mj[j], j] .= rand(lognorm_dist, Mj[j])  
    end

    z_ij .= ( ( 1 - (1-γ)*α )*( (α*(1-γ))/R )^( (α*(1-γ))/(1-α*(1-γ)) ) ).*z_ij_raw.^( 1/(1-α*(1-γ)) )

    return (Mj = Mj, Mj_max = Mj_max, z_ij_raw = z_ij_raw, z_ij = z_ij)

end


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Solve Model
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# 1. Solve for sectoral equilibria
# ------------------------------------------------------------------------------------------------------

function sect_eqlib(param)

    @unpack α, β, η, θ, J, τ, υ, 𝜑, Z_tilde, 𝜑_tilde = param

    # Run productivity simulation
    prod = gen_productivity(param)
    @unpack Mj, Mj_max, z_ij = prod 

    # Initialise 
    what_ij         = zeros(Mj_max,J);      # Wage
    mu_ij           = zeros(Mj_max,J);      # Markdown
    s_ij            = zeros(Mj_max,J);      # Share
    eps_ij          = zeros(Mj_max,J);      # Labor supply elasticity

    # Max iter 
    max_iter = 1000
   
    for j = 1:J
        z_i = z_ij[1:Mj[j], j] 

        # Guess vector for shares 
        s_i = z_i/sum(z_i);     
        
        # Initialize wage, markdown, and elasticity variables 
        w_i = zeros(Mj[j])
        μ_i = zeros(Mj[j])
        eps_i = zeros(Mj[j])
        
        # Initialise 
        iter = 0

        while iter <= max_iter 
            iter += 1    

            # Firm level elasticity of labor supply
            eps_i = ((s_i .* (1/θ) + (1 .- s_i) .* (1/η)).^(-1));

            # Markdown
            μ_i = eps_i ./ (eps_i .+ 1);

            # Wage
            a1 = 1 / (1 + (1 - α) * θ);
            a2 = -(1 - α) * (η - θ) / (η + 1);
            w_i = (μ_i .* α .* z_i .* s_i .^ a2) .^ a1;  

            # Sectoral Wage (CES index)
            W_j               = sum(w_i.^(1+η)).^(1/(1+η));

            # Implied shares
            s_i_new           = (w_i./W_j).^(1+η);  
            
            # Distance of shares and new shares
            dist_S            = maximum(abs.(s_i_new - s_i));

            # Print iteration info for every 500 markets
            if j % 1 == 0 && iter == 1
                println("Processing Market $j... Iterations: ", iter)
            end

            if dist_S < 1.0e-5
                println("Market $j: Shares converged in $iter iterations! Max. % difference = $dist_S")
                break
            end


            # Update S slowly
            s_i               = υ*s_i_new + (1-υ)*s_i;
            s_i               = s_i/sum(s_i); 

        end

        # Store results
        what_ij[1:Mj[j], j] = w_i
        mu_ij[1:Mj[j], j] = μ_i
        s_ij[1:Mj[j], j] = s_i
        eps_ij[1:Mj[j], j] = eps_i
        

    end

    # Compute 'hat' terms 
    what_j = [sum(what_ij[1:Mj[j], j] .^ (η+1)) ^ (1/(η+1)) for j in 1:J]
    What = sum(what_j .^ (η+1)) ^ (1/(η+1))
    nhat_ij = ((what_ij ./ what_j') .^ η) .* ((what_j ./ What) .^ θ)' .* (What .^ 𝜑)

    # Scale up objects
    ω       = Z_tilde / (𝜑_tilde^(1 - α)) 
    W       = ω^(1/(1+(1-α)*𝜑)) * What^( (1+(1-α)*θ) / (1+(1-α) * 𝜑) ) 
    w_ij    = ω^(1/(1+(1-α)*θ)) * W^( ((1-α)*(θ - 𝜑)) / (1 + (1-α) * θ) ) .* what_ij
    w_j     = [sum(w_ij[1:Mj[j], j] .^ (η+1)) ^ (1/(η+1)) for j in 1:J]
    n_ij    = 𝜑_tilde.* ((w_ij ./ w_j') .^ η) .* ((w_j./ W) .^ θ)' .* (W.^ 𝜑)
    n_j     = [sum(n_ij[1:Mj[j], j].^((η+1)/η)).^(η/(η+1)) for j in 1:J]
    N       = sum(n_j.^((θ+ 1 )/ θ)).^(θ / (θ + 1))

    println("Equilibrium computation completed for all $J markets!")
    return (z_ij = z_ij, w_ij = w_ij, n_ij = n_ij, mu_ij = mu_ij, eps_ij = eps_ij, s_ij = s_ij, Mj = Mj)


end

# Run loop
solve_model = sect_eqlib(param)
@unpack z_ij, w_ij, n_ij, mu_ij, s_ij, eps_ij, Mj = solve_model
@unpack η, θ = param

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Plots
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Base Julia
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Compute reference values for horizontal lines
mu_eta_ref = η / (η + 1)
mu_theta_ref = θ / (θ + 1)

# Find the market with exactly 1 firm
market_with_one_firm = findfirst(x -> x == 1, Mj)

# Find a market with more than one firm (choosing the first one found)
market_with_multiple_firms = findfirst(x -> x > 1, Mj)

# Ensure both markets exist
if isnothing(market_with_one_firm) || isnothing(market_with_multiple_firms)
    error("Could not find required markets: one with exactly 1 firm and another with more than 1 firm.")
end

# Store selected markets
selected_markets = [market_with_one_firm, market_with_multiple_firms]
market_labels = ["Single-firm market", "Multi-firm market"]
colors = [:blue, :red]  # Define colors for different markets

# Define new x-axis categorical labels (More Percentiles)
percentile_labels = ["10th", "25th", "50th", "75th", "90th"]
x_positions = 1:length(percentile_labels)  # X positions for categorical axis

# Create subplots
p_s  = plot(xticks=(x_positions, percentile_labels), ylabel="Wage-Bill Share (s_ij)", xlabel="z_ij", 
            title="", legend=:topright)

p_w  = plot(xticks=(x_positions, percentile_labels), ylabel="Wages (w_ij)", xlabel="z_ij", 
            title="", legend=:topright)

p_eps = plot(xticks=(x_positions, percentile_labels), ylabel="Labor Elasticity (eps_ij)", xlabel="z_ij", 
            title="", legend=:topright)

p_mu  = plot(xticks=(x_positions, percentile_labels), ylabel="Markdown (mu_ij)", xlabel="z_ij", 
            title="", legend=:topright)

# Add horizontal dashed lines to p_mu
hline!(p_mu, [mu_eta_ref], linestyle=:dash, label=L"\eta/(\eta+1)", color=:black)
hline!(p_mu, [mu_theta_ref], linestyle=:dash, label=L"\theta/(\theta+1)", color=:gray)

for (i, j) in enumerate(selected_markets)
    # Extract productivity and corresponding y-axis values
    z_values = z_ij[1:Mj[j], j]
    s_values = s_ij[1:Mj[j], j]
    w_values = w_ij[1:Mj[j], j]
    eps_values = eps_ij[1:Mj[j], j]
    mu_values = mu_ij[1:Mj[j], j]

    if Mj[j] == 1
        # Only plot the 10th percentile for the single-firm market
        scatter!(p_s, [x_positions[1]], [s_values[1]], label=market_labels[i], markersize=6, color=colors[i])
        scatter!(p_w, [x_positions[1]], [w_values[1]], label="", markersize=6, color=colors[i])
        scatter!(p_eps, [x_positions[1]], [eps_values[1]], label="", markersize=6, color=colors[i])
        scatter!(p_mu, [x_positions[1]], [mu_values[1]], label="", markersize=6, color=colors[i])
    else
        # Compute percentiles of productivity (5th, 10th, 25th, 50th, 75th, 90th, 95th)
        percentiles = quantile(z_values, [0.10, 0.25, 0.50, 0.75, 0.90])

        # Find corresponding values for percentiles
        s_percentiles = [s_values[argmin(abs.(z_values .- p))] for p in percentiles]
        w_percentiles = [w_values[argmin(abs.(z_values .- p))] for p in percentiles]
        eps_percentiles = [eps_values[argmin(abs.(z_values .- p))] for p in percentiles]
        mu_percentiles = [mu_values[argmin(abs.(z_values .- p))] for p in percentiles]

        # Line plot with dots of the same color
        plot!(p_s, x_positions, s_percentiles, label=market_labels[i], lw=2, color=colors[i])
        scatter!(p_s, x_positions, s_percentiles, label="", markersize=6, color=colors[i])

        plot!(p_w, x_positions, w_percentiles, label="", lw=2, color=colors[i])
        scatter!(p_w, x_positions, w_percentiles, label="", markersize=6, color=colors[i])

        plot!(p_eps, x_positions, eps_percentiles, label="", lw=2, color=colors[i])
        scatter!(p_eps, x_positions, eps_percentiles, label="", markersize=6, color=colors[i])

        plot!(p_mu, x_positions, mu_percentiles, label="", lw=2, color=colors[i])
        scatter!(p_mu, x_positions, mu_percentiles, label="", markersize=6, color=colors[i])
    end
end

# 2x2 grid layout
plot(p_w, p_s, p_eps, p_mu, layout=(2,2), size=(900,700))


# PGFPlotsX for LaTeX equivalent 
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

using PGFPlotsX

# Function for extractions 
function extract_percentiles(z_values, y_values)
    percentiles = quantile(z_values, [0.10, 0.25, 0.50, 0.75, 0.90])
    y_percentiles = [y_values[argmin(abs.(z_values .- p))] for p in percentiles]
    return percentiles, y_percentiles
end

# Recompute the correct percentile values for each selected market
pgf_data = Dict()

for (i, j) in enumerate(selected_markets)
    z_values = z_ij[1:Mj[j], j]
    if Mj[j] == 1

        # Only plot the 10th percentile for the single-firm market
        pgf_data[j] = Dict(
            :w   => [w_ij[1, j]],
            :s   => [s_ij[1, j]],
            :eps => [eps_ij[1, j]],
            :mu  => [mu_ij[1, j]],
            :n   => [n_ij[1, j]]
        )
    else
        # Extract full percentiles for multi-firm markets
        pgf_data[j] = Dict(
            :w   => extract_percentiles(z_values, w_ij[1:Mj[j], j])[2],
            :s   => extract_percentiles(z_values, s_ij[1:Mj[j], j])[2],
            :eps => extract_percentiles(z_values, eps_ij[1:Mj[j], j])[2],
            :mu  => extract_percentiles(z_values, mu_ij[1:Mj[j], j])[2],
            :n   => extract_percentiles(z_values, n_ij[1:Mj[j], j])[2]
        )
    end
end

# Define x-axis labels and colors
percentile_labels = ["10th", "25th", "50th", "75th", "90th"]
x_positions = 1:length(percentile_labels)
colors = ["blue", "red"]

# Plot Function 
function pgf_plot(title, ylabel, key)
    return @pgf Axis(
        {
            title = title,
            xlabel = raw"Productivity percentile, $z_{ij}$",
            ylabel = ylabel,
            xtick = x_positions,
            xticklabels = percentile_labels,
            ymajorgrids = true,
            xmajorgrids = true,
            grid_style = "{opacity=0.2}",
            legend_pos = "south east",
            width="12cm", height="9cm"
        },
        [
            # Single-firm market: Plot only at the first x-position with a solid dot
            Plot({solid, only_marks, mark="*", mark_options="{scale=1.3, fill=$(colors[i])}"}, 
                 Coordinates([(x_positions[1], pgf_data[j][key][1])])) 
            for (i, j) in enumerate(selected_markets) if Mj[j] == 1
        ],
        [
            # Multi-firm market: Plot full percentiles with solid dots
            Plot({solid, color = "red", mark="*", mark_options="{scale=1.3, fill=$(colors[i])}", line_width = "1.1pt"}, 
                 Coordinates(zip(x_positions, pgf_data[j][key]))) 
            for (i, j) in enumerate(selected_markets) if Mj[j] > 1
        ],
        [
            # Single-firm market: Add a horizontal dashed line from the first x-position to max(x_positions)
            Plot({color="blue", dashed, line_width="1pt"}, 
                 Coordinates([(x_positions[1], pgf_data[j][key][1]), 
                              (maximum(x_positions), pgf_data[j][key][1])])) 
            for (i, j) in enumerate(selected_markets) if Mj[j] == 1
        ],

        # Add legend entries
        LegendEntry("Single-firm market"),
        LegendEntry("Multi-firm market")
    )
end


# 1. Wage Plot
# --------------------------------------------------------
pgf_wages = pgf_plot(raw"A. Wages: $w_{ij}$", raw"", :w)
pgfsave("wages_plot.tex", pgf_wages)

# 2. Employment Plot 
# -------------------------------------------------------------------
pgf_emp = pgf_plot(raw"B. Employment: $n_{ij}(w_{ij})$", raw"", :n)
pgfsave("emp_plot.tex", pgf_emp)

# 3. Wage-Bill Share Plot 
# ------------------------------------------------------------------------------------
pgf_wage_bill_share = pgf_plot(raw"B. Wage Payment Share: $s_{ij}^{wn}$", raw"", :s)
pgfsave("wage_bill_share_plot.tex", pgf_wage_bill_share)

# 4. Labor Elasticity Plot 
# ------------------------------------------------------------------------------------------------------------
pgf_labor_elasticity = pgf_plot(raw"C. Labor Supply Elasticity: $\epsilon_{ij}(s_{ij}^{wn})$", raw"", :eps)
pgfsave("labor_elasticity_plot.tex", pgf_labor_elasticity)

# 5. Markdown Plot 
# --------------------------
pgf_markdown = @pgf Axis( 
    {
        title = raw"D. Markdown: $\mu_{ij} = \frac{\epsilon_{ij}(s_{ij}^{wn})}{\epsilon_{ij}(s_{ij}^{wn}) + 1}$",
        xlabel = raw"Productivity percentile, $z_{ij}$",
        ylabel = raw"",
        xtick = x_positions,
        xticklabels = percentile_labels,
        ymajorgrids = true,
        xmajorgrids = true,
        grid_style = "{opacity=0.2}",
        ytick = [mu_eta_ref, mu_theta_ref],  
        ymax = mu_eta_ref + 0.03,
        yticklabels = [raw"$\frac{\eta}{\eta + 1}$", raw"$\frac{\theta}{\theta + 1}$"], 
        legend_pos = "north east",
        width="12cm", height="9cm"
    },
    [
        # Single-firm market: Only plot at the 10th percentile with a solid dot
        Plot({solid, only_marks, mark="*", mark_options="{scale=1.3, fill=$(colors[i])}"}, 
             Coordinates([(x_positions[1], pgf_data[j][:mu][1])])) 
        for (i, j) in enumerate(selected_markets) if Mj[j] == 1
    ],
    [
        # Multi-firm market: Full percentile range with solid dots and thicker line
        Plot({solid, color = "red", mark="*", mark_options="{scale=1.3, fill=$(colors[i])}", line_width="1.1pt"}, 
             Coordinates(zip(x_positions, pgf_data[j][:mu]))) 
        for (i, j) in enumerate(selected_markets) if Mj[j] > 1
    ],

    Plot({color="blue", thick, dashed}, Coordinates([(minimum(x_positions), mu_theta_ref), 
                                                     (maximum(x_positions), mu_theta_ref)])),  


    # Add legend entries
    LegendEntry("Single-firm market"),
    LegendEntry("Multi-firm market")
)


pgfsave("markdown_plot.tex", pgf_markdown)



























































