module TrajUtils
using LinearAlgebra

include("structs.jl")
include("discretize.jl")

export ptr, discretize!, RK4
end
