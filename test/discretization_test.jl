@testset "Discretization Test" begin
    # Pendulum Dynamics and Jacobians
    w = 2 * pi
    function f(x, u)
        return [x[2]; -w^2 * sin(x[1]) + u[1]]
    end
    function dfx(x, u)
        return [0 1; -w^2 * cos(x[1]) 0]
    end
    function dfu(x, u)
        return [0; 1]
    end
    function df(τ::Float64, z::Array{Float64,1}, p::TrajUtils.ptr)
        # Function for integrator
        k = Int(floor(τ / p.dτ)) + 1
        lm = (k * p.dτ - τ) / p.dτ
        lp = (τ - (k - 1) * p.dτ) / p.dτ
        if (k == p.K)
            u_ = u[:, k]
        else
            u_ = lm * u[:, k] + lp * u[:, k+1]
        end
        return f(z, u_)
    end

    nx = 2
    nu = 1
    K = 11
    Nsub = 100
    p = TrajUtils.ptr(nx, nu, K, f, dfx, dfu, :foh, :single)
    p.Nsub = Nsub

    x0 = [0.1; 0.0]
    u = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.0, 2.0]'
    p.uref = u

    xd = zeros(nx, K)
    xc = zeros(nx, K)
    xd[:, 1] = x0
    xc[:, 1] = x0
    TrajUtils.discretize!(p)
    for i = 1:K-1
        xd[:, i+1] = p.A[:, :, i] * xd[:, i] + p.Bm[:, :, i] * u[:, i] + p.Bp[:, :, i] * u[:, i+1]
        xc[:, i+1] = TrajUtils.RK4(df, xc[:, i], (i - 1) * p.dτ, p.dτ, Nsub, p)
    end
    eps = 5e-3
    @test norm(xd-xc) < eps
end
