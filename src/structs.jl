struct IDX

    # Integration state for discretization
    N::Int64                # Number of states in discretization integration state
    x::UnitRange{Int64}     # Indices of state
    phi::UnitRange{Int64}   # Indices of state transition matrix
    Bm::UnitRange{Int64}    # Indices of Bm
    Bp::UnitRange{Int64}    # Indices of Bp
    S::UnitRange{Int64}     # Indices of S
    z::UnitRange{Int64}     # Indices of z

    function IDX(nx, nu)
        idx_x = 1:nx
        idx_phi = (nx+1):(nx+nx^2)
        idx_Bm = (nx+nx^2+1):(nx+nx^2+nx*nu)
        idx_Bp = (nx+nx^2+nx*nu+1):(nx+nx^2+nx*nu+nx*nu)
        idx_S = (nx+nx^2+2*nx*nu+1):(nx+nx^2+2*nx*nu+nx)
        idx_z = (2*nx+nx^2+2*nx*nu+1):(2*nx+nx^2+2*nx*nu+nx)

        new(nx^2 + 3 * nx + 2 * nx * nu, idx_x, idx_phi, idx_Bm, idx_Bp, idx_S, idx_z)
    end

end
mutable struct ptr

    # Problem parameters
    nx::Int64     # Number of states
    nu::Int64     # Number of controls
    K::Int64      # Number of PTR nodes
    dτ::Float64   # Normalized time per interval

    # PTR Hyperparameters
    Nsub::Int64   # Number of integration subintervals for discretization
    wtr::Float64   # Weight on trust region
    wvc::Float64  # Virtual control weight
    wvb::Float64  # Virtual buffer weight

    # Dynamics and Jacobians
    f::Function   # CT Dynamics
    dfx::Function # State Jacobian
    dfu::Function # Control Jacobian

    # Reference Trajectories
    xref::Array{Float64,2} # Reference state trajectory
    uref::Array{Float64,2} # Reference control trajectory
    σref                   # Reference time dilation

    # Integrated Trajectory
    xint::Array{Float64,2}

    # Virtual Control and Virtual Buffer
    vc::Array{Float64,2}
    vb::Array{Float64,1}

    # Trust region Radius
    Δ::Float64
    Δσ::Float64

    # Discrete Dynamics
    idx::IDX
    xprop::Array{Float64,2}  # Propagated state
    def::Array{Float64,1}    # Defect
    A::Array{Float64,3}      # Discrete A matrices
    Bm::Array{Float64,3}     # Discrete Bm matrices
    Bp::Array{Float64,3}     # Discrete Bp matrices
    S::Array{Float64,2}      # Discrete S vectors
    z::Array{Float64,2}      # Discrete z vector 

    # Problem paramters
    # par::PARAMS

    # Discretiation Type
    disc::Symbol             # Either :foh or :impulsive

    # Dilation Type
    dilation::Symbol         # Either :single or :multiple

    function ptr(nx::Int64, nu::Int64, K::Int64, f::Function, dfx::Function, dfu::Function, disc::Symbol, dilation::Symbol)
        Nsub = 10
        wtr = 0.2
        wvc = 13.0
        wvb = 0.1

        if (disc != :foh && disc != :impulsive)
            throw(ArgumentError("disc must be :foh or :impulsive"))
        end

        if (dilation != :single && dilation != :multiple)
            throw(ArgumentError("dilation must be :single or :multiple"))
        end

        if (dilation == :single)
            σref = 1.0
            dτ = 1 / (K - 1)
        elseif (dilation == :multiple)
            σref = ones(K - 1)
            dτ = 1.0
        end

        new(nx, nu, K, dτ, Nsub, wtr, wvc, wvb, f, dfx, dfu, zeros(nx, K), zeros(nu, K), σref, zeros(nx, K + (K - 1) * (Nsub - 1)), zeros(nx, K - 1), zeros(K), 0.0, 0.0, IDX(nx, nu), zeros(nx, K - 1), zeros(K - 1), zeros(nx, nx, K - 1), zeros(nx, nu, K - 1), zeros(nx, nu, K - 1), zeros(nx, K - 1), zeros(nx, K - 1), disc, dilation)
    end
end