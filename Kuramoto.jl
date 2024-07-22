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
    t = range(start=tmin,stop=tmax,step=Δt);
    L = length(t);

    parameters = (N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian) 
    return parameters
end

function create_network(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian = parameters;
    G = Graphs.SimpleGraphs.erdos_renyi(N,Float32(p)) 
    A = Graphs.sparse(G);
    return A
end

function create_initial_condition(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian = parameters
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
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian = parameters
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

# Input from command line
path = ARGS[1]
index = parse(Int,ARGS[2])

# Hyper-parameters
parameters = create_parameters(path,index);
iteration_network = parameters[10]
iteration_brownian = parameters[11]
tot = iteration_brownian*iteration_network


# Main part
parameters = create_parameters(path,index)#path,index);
r = zeros(length(parameters[6]))

for itNet in 1:iteration_network
    A = create_network(parameters);
    for itBrown in 1:iteration_brownian
        x0 = create_initial_condition(parameters);
        r .+=  integrate_N_particle_system(parameters,x0,A) ./tot
    end
end

# Saving the data
JLD2.jldsave("Data.jld2"; order_parameter = r, parameters)