#### LIBRARIES AND FUNCTIONS
using Plots
using Graphs#, GraphRecipes
using LinearAlgebra
using StatsBase
using DataFrames
using CSV
using JLD2


########### CHANGE THIS FUNCTION TO SPECIFY THE COEFFICIENTS/HARMONICS OF THE INTERACTION ########

function create_coefficients()
    # We define here the interaction potential W(x) = ∑c_n sin(nx)
    n = [1]
    c_n = [1]    
    return n,c_n
end


############ THESE SHOULD NOT BE EDITED ###########
function create_parameters(path,index)
    P = DataFrame(CSV.File("parameters.csv"));
    # Parameters for the network
    N = P[!,"N"][index]
    p = P[!,"p"][index]
    # Parameter of the system
    θ = P[!,"theta"][index]
    σ = P[!,"sigma"][index]
    # Parameters for integration
    tmin = P[!,"tmin"][index]; tmax = P[!,"tmax"][index]; Δt = P[!,"Dt"][index]

    it_network = P[!,"it_network"][index]; it_brownian = P[!,"it_brownian"][index]

    p_WS = P[!,"p_WS"][index] ; r = P[!,"r"][index]

    t = range(start=tmin,stop=tmax,step=Δt);
    L = length(t);

    parameters = (N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS) 
    return parameters
end

function create_network_ER(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS = parameters;
    G = Graphs.SimpleGraphs.erdos_renyi(N,Float32(p)) 
    A = Graphs.sparse(G);
    return A
end

function create_network_small_world(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS = parameters;
    G = Graphs.SimpleGraphs.watts_strogatz(N,r,p_WS) 
    A = Graphs.sparse(G);
    return A
end

function create_network_ring(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS = parameters;
    G = Graphs.SimpleGraphs.watts_strogatz(N,r,0) 
    A = Graphs.sparse(G);
    return A
end

function create_initial_condition(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS = parameters
    x0 = mod.( 2*π*rand(N) , 2*π);
    return x0
end

function coupling_interaction(x,n,c_n)
    L = length(n)
    a = 0 
    for i in 1:L
        a += c_n[i] *n[i] * sin(n[i]*x)
    end
    return a
end

function order_param(x,n)
    N = size(x)
    L = length(n)

    r = zeros(ComplexF64,L,1)
    for i in 1:L
        r[i] = mean( exp.(1im*n[i]*x))
    end 
    
    return abs.(r)
end

function get_energy(r,c_n)
    L = length(n)
    U = 0
    for i in 1:L
        U += -1/2*c_n[i]*r[i]^2
    end
    return U
end

function integrate_N_particle_system(parameters,x0,A,n,c_n)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS = parameters
    L_n = length(n)
    xnew = zeros(N) ; xold = copy(x0); r = zeros(L_n,L); U = zeros(L)
    for tt=1:L
        for i=1:N
            coupl = coupling_drift.(xold[i] .- xold)
            xnew[i] = mod( xold[i] - θ/N * dot(A[:,i],coupl) + σ * randn() , 2*π)
        end
        OP = order_param(xnew,n)
        r[:,tt] = OP
        U[tt] = get_energy(OP,c_n)
        xold = copy(xnew);
    end
    return r,U
end

function main(parameters)
 
    iteration_network = parameters[10]
    iteration_brownian = parameters[11]
    tot = iteration_brownian*iteration_network

    r = Array{Matrix{Float64}}(undef, tot)#zeros(tot,length(parameters[6]))
    Energy = zeros(tot,length(parameters[6]))

    index = 1
    for itNet in 1:iteration_network
        A = create_network_ring(parameters);
        for itBrown in 1:iteration_brownian
            x0 = create_initial_condition(parameters);
            R,u = integrate_N_particle_system(parameters,x0,A,n,c_n)

            r[index] = R
            Energy[index,:] = u

            index += 1
        end
    end

    return r, Energy
end

# Input from command line
path = "./parameters.csv"#ARGS[1]
index = 1#parse(Int,ARGS[2])
# Load parameters and dynamical specification of the system (harmonics of the potential)
parameters = create_parameters(path,index);
n , c_n = create_coefficients()
coupling_drift(x) = coupling_interaction(x,n,c_n)

# Main Function
r , Energy =  main(parameters)

JLD2.jldsave("./data/data"*string(index)*"/Data.jld2"; order_parameter = r, Energy = Energy ,parameters)