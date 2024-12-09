using Random; Random.seed!(984519674645)
using StableRNGs; rng = StableRNG(845652695)
include("definitions.jl")

# first experiment

optimization_state =  zeros(15)
optimization_initial = optimization_state[1]
@mtkbuild true_bioreactor = TrueBioreactor()
prob = ODEProblem(true_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
sol = solve(prob, Rodas5P())

@mtkbuild  ude_bioreactor = UDEBioreactor()
ude_prob = ODEProblem(ude_bioreactor, [], (0.0, 15.0), [], tstops = 0:15, save_everystep=false)
ude_sol = solve(ude_prob, Rodas5P())

data = DataFrame(sol)
data = data[1:2:end, :] # HACK TO GET ONLY THE MEASUREMENTS WE NEED; MTK ALWAYS SAVES BEFORE AND AFTER CALLBACK; WITH NO OPTION TO DISABLE

data[!, "C_s(t)"] += sd_cs * randn(size(data, 1))

of = OptimizationFunction{true}(loss, AutoZygote())
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))
get_vars = getu(ude_bioreactor, [ude_bioreactor.C_s])
ps = ([ude_prob], [get_vars], [data]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=1000)

new_p = SciMLStructures.replace(Tunable(), ude_prob.p, res.u)
res_prob = remake(ude_prob, p=new_p)
res_sol = solve(res_prob, Rodas5P())

extracted_chain = arguments(equations(ude_bioreactor.nn)[1].rhs)[1]
T = defaults(ude_bioreactor)[ude_bioreactor.nn.T]
C_s_range_plot = 0.0:0.01:50.0
μ_predicted_plot = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]
μ_predicted_data = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]

## get plausible model structures for missing physics

hall_of_fame = equation_search(collect(data[!, "C_s(t)"])', μ_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)
    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)
end
plot!(plts[1], sol, idxs=:C_s, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)
plot!(plts[1], ylabel="Cₛ(g/L)", xlabel="t(h)")
scatter!(plts[1], data[!, "timestamp"], data[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol, idxs=:C_x, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)
plot!(plts[2], ylabel="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol, idxs=:V, ylabel="V(L)", xlabel="t(h)", lw=3, color=:black, ylims=(6.0,8.0))
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], C_s_range_plot, μ_predicted_plot, lw=3, c=2)
scatter!(plts[4], data[!, "C_s(t)"], μ_predicted_data, ms=3, c=2)
plot!(plts[4], ylabel="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("experiment1.pdf")

# optimize the control pars
design_prob = OptimizationProblem(S_criterion, optimization_state, (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(design_prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=100.0)

optimization_state = control_pars_opt.u
optimization_initial = optimization_initial2 = optimization_state[1]

plts = plot(), plot()
t_pwc = []
pwc = []
for i in 0:14
    push!(t_pwc,i)
    push!(t_pwc,i+1)
    push!(pwc,optimization_state[i+1])
    push!(pwc,optimization_state[i+1])
end
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t(h)",ylabel="Qin(L/h)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, C_s = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    plot!(plts[2], sol_plausible, idxs=:C_s, lw=3,ls=:dash,c=i+2)
end
plot!(plts[2],xlabel="t(h)",ylabel="Cₛ(g/L)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("control1.pdf")

# second experiment
@mtkbuild true_bioreactor2 = TrueBioreactor()
prob2 = ODEProblem(true_bioreactor2, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol2 = solve(prob2, Rodas5P())
@mtkbuild ude_bioreactor2 = UDEBioreactor()
ude_prob2 = ODEProblem(ude_bioreactor2, [], (0.0, 15.0), [ude_bioreactor2.Q_in => optimization_initial], tstops=0:15, save_everystep=false)
ude_sol2 = solve(ude_prob2, Rodas5P())
plot(ude_sol2[3,:])
ude_prob_remake = remake(ude_prob, p=ude_prob2.p)
sol_remake = solve(ude_prob_remake, Rodas5P())
plot(sol_remake[3,:])
x0 = reduce(vcat, getindex.((default_values(ude_bioreactor),), tunable_parameters(ude_bioreactor)))

get_vars2 = getu(ude_bioreactor2, [ude_bioreactor2.C_s])

data2 = DataFrame(sol2)
data2 = data2[1:2:end, :]
data2[!, "C_s(t)"] += sd_cs * randn(size(data2, 1))

ps = ([ude_prob, ude_prob2], [get_vars, get_vars2], [data, data2]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, NLopt.LN_BOBYQA, maxiters=5_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob2.p, res.u)
res_prob = remake(ude_prob2, p=new_p)
callback_controls, initial_control, C_s = syms_cache[1]
res_prob.ps[initial_control] = optimization_initial2
res_sol = solve(res_prob, Rodas5P())
## get chain from the equations
extracted_chain = arguments(equations(ude_bioreactor2.nn)[1].rhs)[1]
T = defaults(ude_bioreactor2)[ude_bioreactor2.nn.T]
μ_predicted_plot2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]

μ_predicted_data = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]

total_data = hcat(collect(data[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data, μ_predicted_data2)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plts = plot(), plot(), plot(), plot()
for i in 1:length(model_structures)
    plot!(plts[4],  C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)
    plausible_prob = probs_plausible[i]
    sol_plausible = solve(plausible_prob, Rodas5P())
    # plot!(sol_plausible; label=["Cₛ(g/L)" "Cₓ(g/L)" "V(L)"], xlabel="t(h)", lw=3)
    plot!(plts[1], sol_plausible, idxs=:C_s, lw=1,ls=:dash,c=i+2)
    plot!(plts[2], sol_plausible, idxs=:C_x, lw=1,ls=:dash,c=i+2)
end
plot!(plts[1], sol2, idxs=:C_s, lw=3,c=1)
plot!(plts[1], res_sol, idxs=:C_s, lw=3,c=2)
plot!(plts[1], ylabel="Cₛ(g/L)", xlabel="t(h)")
scatter!(plts[1], data2[!, "timestamp"], data2[!, "C_s(t)"]; ms=3,c=1)
plot!(plts[2], sol2, idxs=:C_x, lw=3,c=1)
plot!(plts[2], res_sol, idxs=:C_x, lw=3,c=2)
plot!(plts[2], ylabel="Cₓ(g/L)", xlabel="t(h)")
plot!(plts[3], sol2, idxs=:V, ylabel="V(L)", xlabel="t(h)", lw=3, color=:black)
plot!(plts[4], C_s_range_plot, μ_max .* C_s_range_plot ./ (K_s .+ C_s_range_plot), lw=3, c=1)
plot!(plts[4], C_s_range_plot, μ_predicted_plot2, lw=3, c=2)
scatter!(plts[4], data[!, "C_s(t)"], μ_predicted_data, ms=3, c=2)
scatter!(plts[4], data2[!, "C_s(t)"], μ_predicted_data2, ms=3, c=2)
plot!(plts[4], ylabel="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5))
plot(plts..., layout = 4, tickfontsize=10, guidefontsize=12, legendfontsize=14, grid=false, legend=false)
savefig("experiment2.pdf")

S_criterion(zeros(15), (probs_plausible, syms_cache))

prob = OptimizationProblem(S_criterion, zeros(15), (probs_plausible, syms_cache), lb=lb, ub=ub)
control_pars_opt = solve(prob, BBO_adaptive_de_rand_1_bin_radiuslimited(), maxtime=60.0)

optimization_state = control_pars_opt.u
optimization_initial = optimization_state[1]

plts = plot(), plot()
t_pwc = []
pwc = []
for i in 0:14
    push!(t_pwc,i)
    push!(t_pwc,i+1)
    push!(pwc,optimization_state[i+1])
    push!(pwc,optimization_state[i+1])
end
plot!(plts[1], t_pwc, pwc, lw=3, color=:black,xlabel="t(h)",ylabel="Qin(L/h)")
for i in 1:length(model_structures)
    plausible_prob = probs_plausible[i]
    callback_controls, initial_control, C_s = syms_cache[i]
    plausible_prob.ps[callback_controls] = control_pars_opt[2:end]
    plausible_prob.ps[initial_control] = control_pars_opt[1]
    sol_plausible = solve(plausible_prob, Rodas5P())
    plot!(plts[2], sol_plausible, idxs=:C_s, lw=3,ls=:dash,c=i+2)
end
plot!(plts[2],xlabel="t(h)",ylabel="Cₛ(g/L)")
plot(plts..., layout = (2, 1), tickfontsize=12, guidefontsize=14, legendfontsize=14, grid=false, legend=false)
savefig("control2.pdf")


# third experiment
@mtkbuild true_bioreactor3 = TrueBioreactor()
prob3 = ODEProblem(true_bioreactor3, [], (0.0, 15.0), [], tstops=0:15, save_everystep=false)
sol3 = solve(prob3, Rodas5P())
@mtkbuild ude_bioreactor3 = UDEBioreactor()
ude_prob3 = ODEProblem(ude_bioreactor3, [], (0.0, 15.0), tstops=0:15, save_everystep=false)

x0 = reduce(vcat, getindex.((default_values(ude_bioreactor3),), tunable_parameters(ude_bioreactor3)))

get_vars3 = getu(ude_bioreactor3, [ude_bioreactor3.C_s])

data3 = DataFrame(sol3)
data3 = data3[1:2:end, :]
data3[!, "C_s(t)"] += sd_cs * randn(size(data3, 1))

ps = ([ude_prob, ude_prob2, ude_prob3], [get_vars, get_vars2, get_vars3], [data, data2, data3]);
op = OptimizationProblem(of, x0, ps)
res = solve(op, Optimization.LBFGS(), maxiters=10_000)

new_p = SciMLStructures.replace(Tunable(), ude_prob3.p, res.u)
res_prob = remake(ude_prob3, p=new_p)
res_sol = solve(res_prob, Rodas5P())
plot(res_sol)
plot!(sol3)

## get chain from the equations
extracted_chain = arguments(equations(ude_bioreactor3.nn)[1].rhs)[1]
T = defaults(ude_bioreactor3)[ude_bioreactor3.nn.T]

μ_predicted_data = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data[!, "C_s(t)"]]
μ_predicted_data2 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data2[!, "C_s(t)"]]
μ_predicted_data3 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in data3[!, "C_s(t)"]]

total_data = hcat(collect(data[!, "C_s(t)"]'), collect(data2[!, "C_s(t)"]'), collect(data3[!, "C_s(t)"]'))
total_predicted_data =  vcat(μ_predicted_data, μ_predicted_data2, μ_predicted_data3)
hall_of_fame = equation_search(total_data, total_predicted_data; options, niterations=1000, runtests=false, parallelism=:serial)

scatter(total_data', total_predicted_data,legend=false)

model_structures = get_model_structures(hall_of_fame, options, n_best)
probs_plausible, syms_cache = get_probs_and_caches(model_structures);

plot()
μ_predicted_plot3 = [only(LuxCore.stateless_apply(extracted_chain, [C_s], convert(T,res.u))) for C_s in C_s_range_plot]
for i in 1:length(model_structures)
    plot!(C_s_range_plot, model_structures[i].( C_s_range_plot);c=i+2,lw=1,ls=:dash)
end
plot!(C_s_range_plot, μ_predicted_plot3, lw=3, c=2)
scatter!(data[!, "C_s(t)"], μ_predicted_data, ms=3, c=2)
scatter!(data2[!, "C_s(t)"], μ_predicted_data2, ms=3, c=2)
scatter!(data3[!, "C_s(t)"], μ_predicted_data3, ms=3, c=2)
plot!(ylabel="μ(1/h)", xlabel="Cₛ(g/L)",ylims=(0,0.5),legend=false)
