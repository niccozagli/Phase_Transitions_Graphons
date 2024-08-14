#### LIBRARIES AND FUNCTIONS
using Plots
using Graphs#, GraphRecipes
using LinearAlgebra
using StatsBase
using DataFrames
using CSV
using JLD2

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

function coupling_drift(x)
    return sin(x)
end

function order_param(x)
    N = size(x)
    r = mean( exp.(1im*x) )
    return abs(r)
end

function integrate_N_particle_system(parameters,x0,A)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS = parameters
    xnew = zeros(N) ; xold = copy(x0); r = zeros(L)
    for tt=1:L
        for i=1:N
            coupl = coupling_drift.(xold[i] .- xold)
            xnew[i] = mod( xold[i] - θ/N * dot(A[:,i],coupl) + σ * randn() , 2*π)
        end
        r[tt] = order_param(xnew)
        xold = copy(xnew);
    end
    return r
end

function main(parameters)
    iteration_network = parameters[10]
    iteration_brownian = parameters[11]
    tot = iteration_brownian*iteration_network
    r = zeros(tot,length(parameters[6]))
    index = 1
    for itNet in 1:iteration_network
        A = create_network_ring(parameters);
        for itBrown in 1:iteration_brownian
            x0 = create_initial_condition(parameters);
            r[index,:] = integrate_N_particle_system(parameters,x0,A)
            index += 1
            println(index)
        end
    end
    return r
end

# Input from command line
path = ARGS[1]
index = parse(Int,ARGS[2])
parameters = create_parameters(path,index);

r = main(parameters)

# Saving the data
JLD2.jldsave("Data.jld2"; order_parameter = r, parameters)