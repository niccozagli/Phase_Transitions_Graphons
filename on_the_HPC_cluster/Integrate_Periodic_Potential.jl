#### LIBRARIES AND FUNCTIONS
using Graphs #, GraphRecipes
using LinearAlgebra
using StatsBase
using DataFrames
using CSV
using JLD2


########### CHANGE THIS FUNCTION TO SPECIFY THE COEFFICIENTS/HARMONICS OF THE INTERACTION ########

function create_coefficients()
    # We define here the interaction potential W(x) = ∑c_n sin(nx)
    n = [1,2]
    c_n = [1,2]    
    return n,c_n
end


############ THESE SHOULD NOT BE EDITED ###########
function create_parameters(path,index)
    P = DataFrame(CSV.File(path));
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

    t_integration = tmin:Δt:tmax;   #range(start=tmin,stop=tmax,step=Δt);
    tau = 10
    t = tmin:Δt:tmax;
    L = length(t);

    parameters = (N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS,tau) 
    return parameters
end

function create_network_ER(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS,tau = parameters
    G = Graphs.SimpleGraphs.erdos_renyi(N,Float32(p)) 
    A = Graphs.sparse(G);
    factor = 1
    return A , factor
end

function create_network_small_world(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS,tau = parameters;
    G = Graphs.SimpleGraphs.watts_strogatz(N,r,p_WS) 
    A = Graphs.sparse(G);
    factor = 1
    return A , factor
end

function create_network_ring(parameters)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS,tau = parameters;
    G = Graphs.SimpleGraphs.watts_strogatz(N,r,0) 
    A = Graphs.sparse(G);
    factor = 1
    return A , factor 
end

function create_network_power_law(parameters,α,β)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS,tau = parameters
    g = SimpleGraph(N)
    ρn = N^(-β)
    W(x,y) = (x*y)^(-α)

    for i in 1:N
        xi = i/N
        for j in 1:i-1 
            xj = j/N
            w = min( 1/ρn , W(xi,xj) )
            prob = ρn * w 

            if( rand() < prob )
                add_edge!(g,i,j)
            end

        end
    end

    A = Graphs.sparse(g);
    factor = ρn
    return A , factor
end

function create_initial_condition(parameters)
    N = parameters[1]
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


function integrate_N_particle_system(parameters,x0,A,n,c_n,factor)
    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS,tau = parameters
    t_save = tmin : Δt * tau : tmax; L_save = length(t_save)

    coupling_drift(x) = coupling_interaction(x,n,c_n)
    L_n = length(n)
    xnew = zeros(N) ; xold = copy(x0); r = zeros(L_n,L_save-1); U = zeros(L_save-1)
    index = 1
    coupl = zeros(N)
    for tt=1:L
        # for i=1:N
        #     coupl[:] =  coupling_drift.(xold[i] .- xold)
        #     xnew[i] = @views mod( xold[i] - θ/(N*factor) * dot( A[:,i],coupl) + σ * randn() , 2*π)
        # end
        for i=1:N
            for j=1:N
                coupl[j] = coupling_drift.(xold[i] - xold[j])
            end
            xnew[i] = @views mod( xold[i] - θ/(N*factor) * dot( A[:,i],coupl) + σ * randn() , 2*π)
        end


        ## Saving data
        if(mod(tt,tau)==0)
            OP = order_param(xnew,n)
            r[:,index] = OP
            U[index] = get_energy(OP,c_n)
            index += 1
        end

        xold = copy(xnew);
    end
    return r,U
end

function main(parameters,n,c_n)

    N,p,tmin,tmax,Δt,t,L,θ,σ,it_network,it_brownian, r, p_WS,tau = parameters
    #println("tmax = "*string(tmax))
    #println("Blas_threads="*string(BLAS.get_num_threads()))
    #println("julia_threads="*string(Base.Threads.nthreads()))
    tot = it_brownian*it_network
    t_save = tmin:Δt*tau:tmax; 
    r = Array{Matrix{Float64}}(undef, tot)#zeros(tot,length(parameters[6]))
    Energy = zeros(tot,length(t_save)-1)
    
    index = 1
    for itNet in 1:it_network
        α = 0.4; β = 0.48 #### CHANGE HERE THE PARAMETERS FOR THE POWER LAW NETWORK
        A , factor = create_network_power_law(parameters,α,β)
        #A , factor = create_network_small_world(parameters)
        for itBrown in 1:it_brownian
            x0 = create_initial_condition(parameters);
            R,u = integrate_N_particle_system(parameters,x0,A,n,c_n,factor)

            r[index] = R
            Energy[index,:] = u

            index += 1
        end
    end

    return r, Energy
end

# Input from command line
path = ARGS[1]
index = parse(Int,ARGS[2])
# Load parameters and dynamical specification of the system (harmonics of the potential)
parameters = create_parameters(path,index);
n , c_n = create_coefficients()

#BLAS.set_num_threads(1)
#using BenchmarkTools
#@btime main($parameters,$n,$c_n)

# Main Function
r , Energy =  main(parameters,n,c_n)

path_save = "./data/data"*string(index)*"/Data.jld2"
JLD2.jldsave(path_save; order_parameter = r, Energy = Energy ,parameters)