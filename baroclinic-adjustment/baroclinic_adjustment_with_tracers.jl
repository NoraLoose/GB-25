using Oceananigans
using Reactant
using Oceananigans.Units
using Oceananigans.Architectures: ReactantState
using Random
using Printf

# input params
params = Dict(
    "precis" => "Float64", # Float64, Float32
    "resol"  => "1//8",    # 1//8, 1//16
    "zcoord" => "z"    # zstar, z
)
target_dir = "/pscratch/sd/n/nloose/GB-25/baroclinic-adjustment/" # change to where you want save stuff 

precision = eval(Meta.parse(params["precis"]))
resolution = eval(Meta.parse(params["resol"]))

prefix = joinpath(target_dir, "baroclinic_adjustment_$(params["zcoord"])_$(Base.Int(1 // resolution))_$(precision)")

# Set default floating point type
Oceananigans.defaults.FloatType = precision

# Architecture
arch = GPU()  # this script is NOT tested with arch = ReactantState()
Lz = 1kilometers     # depth [m]
Ny = Base.Int(20 / resolution)
Nz = 50
N² = 4e-6  # [s⁻²] buoyancy frequency / stratification
Δb = 0.005 # [m/s²] buoyancy difference
φ₀ = 50

if params["zcoord"] == "zstar"
    z = Oceananigans.Grids.MutableVerticalDiscretization((-Lz, 0))
else
    z = (-Lz, 0)
end
closure = VerticalScalarDiffusivity(precision; κ=1e-5, ν=1e-4)

stop_time = 300days

@info "Nx, Ny, Nz = $Ny, $Ny, $Nz"

grid = LatitudeLongitudeGrid(arch,
                             topology = (Periodic, Bounded, Bounded),
                             size = (Ny, Ny, Nz),
                             longitude = (-10, 10),
                             latitude = (φ₀ - 10, φ₀ + 10),
                             z = z,
                             halo = (6, 6, 6))

# time step
dx = minimum_xspacing(grid)
Δt = 0.15 * dx / 2 # c * dx / max(U)

# Tracer forcing

# Define delta function for tracer release
i0 = Base.Int(Ny / 2)
j0 = Base.Int(Ny / 2)
t0 = 5days - Δt
tf = t0 + Δt
@inline t_unforced(t, t0, tf) = (t < t0) | (t >= tf)
@inline i_unforced(i, i0) = (i < i0) | (i > i0)
@inline j_unforced(j, j0) = (j < j0) | (j > j0)
@inline k_unforced(k, k0) = (k < k0) | (k > k0)
@inline unforced(i, j, k, t, p) = t_unforced(t, p.t0, p.tf) | i_unforced(i, p.i0) | j_unforced(j, p.j0) | k_unforced(k, p.k0)
@inline forcing(i, j, k, grid, clock, fields, p) = ifelse(unforced(i, j, k, clock.time, p), zero(grid), one(grid))

# release at the surface
c1_forcing = Forcing(forcing, discrete_form=true, parameters=(; i0=i0, j0=j0, k0=Nz, t0=t0, tf=tf))
# release at the bottom
c2_forcing = Forcing(forcing, discrete_form=true, parameters=(; i0=i0, j0=j0, k0=1, t0=t0, tf=tf))


# Model
model = HydrostaticFreeSurfaceModel(; grid, closure,
                                    coriolis = HydrostaticSphericalCoriolis(),
                                    buoyancy = BuoyancyTracer(),
                                    tracers = (:b, :c1, :c2),
                                    forcing = (; c1=c1_forcing, c2=c2_forcing),
                                    momentum_advection = WENOVectorInvariant(),
                                    tracer_advection = WENO(order=7))

# Parameters
parameters = (; N², Δb, φ₀, Δφ = 20)

@inline function bᵢ(λ, φ, z, p)
    γ = π/2 - 2π * (p.φ₀ - φ) / p.Δφ
    b = ifelse(γ < 0, 0, ifelse(γ > π, 1, 1 - (π - γ - sin(π - γ) * cos(π - γ)) / π))
    return p.N² * z + p.Δb * b
end

ϵb = 1e-2 * Δb # noise amplitude
Random.seed!(1234)
bᵢ(x, y, z) = bᵢ(x, y, z, parameters) + ϵb * randn()
set!(model, b=bᵢ)


simulation = Simulation(model; Δt, stop_time)

wall_clock = Ref(time_ns())

function progress(sim)

    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   maximum(abs, sim.model.velocities.w))

    @info msg

    wall_clock[] = time_ns()

    return nothing
end

add_callback!(simulation, progress, TimeInterval(10days))

u, v, w = model.velocities
e = @at (Center, Center, Center) (u^2 + v^2) / 2
E = Average(e, dims=(1, 2, 3))
ke_ow = JLD2OutputWriter(model, (; E),
                         filename = prefix * "_kinetic_energy.jld2",
                         schedule = TimeInterval(1days),
                         overwrite_existing = true)
simulation.output_writers[:ke] = ke_ow

c1 = model.tracers.c1
c2 = model.tracers.c2
c1_avg = Average(c1, dims=(1, 2, 3))
c2_avg = Average(c2, dims=(1, 2, 3))
c1_int = Integral(c1)
c2_int = Integral(c2)
c_ow = JLD2OutputWriter(model, (; c1_avg, c2_avg, c1_int, c2_int),
                         filename = prefix * "_tracers.jld2",
                         schedule = TimeInterval(1days),
                         overwrite_existing = true)
simulation.output_writers[:c] = c_ow

Nz = size(grid, 3)
b = model.tracers.b
ζ = ∂x(v) - ∂y(u)
fields = (; u, v, w, b, ζ, c1, c2)
f_ow = JLD2OutputWriter(model, fields,
                        filename = prefix * "_fields.jld2",
                        indices = (:, :, Nz),
                        schedule = TimeInterval(10days),
                        overwrite_existing = true)

simulation.output_writers[:fields] = f_ow

if arch isa ReactantState
    _run! = @compile run!(simulation)
else
    _run! = run!
end

_run!(simulation)
