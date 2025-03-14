using CairoMakie
using Oceananigans
using Printf

function vis_evol(dir; integrals=true, error=true, only_depth_tracer=false)

    t0 = 7 # end of tracer release in days + 1

    pt = 4 / 3
    fig = Figure(; size=(600, 500), fontsize=12pt)

    if integrals
        ylabel = "Integral"
    else
        ylabel = "Average"
    end
    if error
        title = "Relative Error in Tracer $ylabel"
    else
        title = "Tracer $ylabel"
    end
    
    ax = Axis(fig[1, 1], xlabel="Time (days)", ylabel="", title=title)

    for zcoord in ["zstar", "z"]
        for resolution in ["8", "16"]
            for precision in ["Float64", "Float32"]
                filename = joinpath(dir, "baroclinic_adjustment_" * zcoord * "_" * resolution * "_" * precision * "_tracers.jld2")
                
                if isfile(filename)
                    if integrals
                        C1t = FieldTimeSeries(filename, "c1_int")
                        C2t = FieldTimeSeries(filename, "c2_int")
                    else
                        C1t = FieldTimeSeries(filename, "c1_avg")
                        C2t = FieldTimeSeries(filename, "c2_avg") 
                    end

                    label1 = "Surface tracer, $precision, 1/$(resolution)°, $zcoord-coord"
                    label2 = "Depth tracer, $precision, 1/$(resolution)°, $zcoord-coord"

                    times1 = C1t.times[t0:end]/3600/24
                    times2 = C2t.times[t0:end]/3600/24

                    if error
                        data1 = (C1t.data[1, 1, 1, t0:end] .- C1t.data[1, 1, 1, t0]) ./ C1t.data[1, 1, 1, t0]
                        data2 = (C2t.data[1, 1, 1, t0:end] .- C2t.data[1, 1, 1, t0]) ./ C2t.data[1, 1, 1, t0]
                    else
                        data1 = C1t.data[1, 1, 1, t0:end]
                        data2 = C2t.data[1, 1, 1, t0:end]    
                    end

                                        if !only_depth_tracer
                        lines!(ax, times1, data1, linewidth=4, label=label1)
                    end
                    lines!(ax, times2, data2, linewidth=4, label=label2)
                end
            end
        end
    end

    # Add legend
    axislegend(ax, position=:lb)
    return fig
end

#dir = "/pscratch/sd/n/nloose/GB-25/baroclinic-adjustment/"
#fig = vis_evol(dir; integrals=true, error=true)
#fig = vis_evol(dir; integrals=true, error=false)
#fig = vis_evol(dir; integrals=true, error=false, only_depth_tracer=true)