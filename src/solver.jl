const LinsolveSubarray = SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}

function admm_step!(x::Vector{Float64},
	s::SplitVector{Float64},
	μ::Vector{Float64},
	v::Vector{Float64},
	ν::LinsolveSubarray,
	x_tl::LinsolveSubarray,
	s_tl::Vector{Float64},
	ls::Vector{Float64},
	sol::Vector{Float64},
	F,
	q::Vector{Float64},
	b::Vector{Float64},
	ρ::Vector{Float64},
	α::Float64,
	σ::Float64,
	m::Int64,
	n::Int64,
	set::CompositeConvexSet{Float64}, v1::SubArray, v2::SubArray)

	# linear solve
	# Create right hand side for linear system
	# deconstructed solution vector is ls = [x_tl(n+1); ν(n+1)]
	# x_tl and ν are automatically updated, since they are views on sol
	# ρ  = 0.1

	@. ls[1:n] = σ * v1 - q # or can it be ρ?????
	@. ls[(n + 1):end] = b - v2
	sol .= F \ ls

	# Over relaxation
	@. x = 2 * x_tl - v1
	@. s_tl = v2 - ν / ρ

	# Project onto cone
	@. s = 2 * s_tl - v2
	p_time = @elapsed project!(s, set)

	# recover original dual variable for conic constraints
	@. μ = ρ * (s - v2)

	# update dual variable v
	@. v[1:n] = v[1:n] + 2 * α .* ( x - x_tl)
	@. v[n+1:n+m] = v[n+1:n+m] + 2 * α .* ( s - s_tl)



	return p_time
end

# SOLVER ROUTINE
# -------------------------------------


"""
optimize!(model)

Attempts to solve the optimization problem defined in `COSMO.Model` object with the user settings defined in `COSMO.Settings`. Returns a `COSMO.Result` object.
"""
function optimize!(ws::COSMO.Workspace)
	solver_time_start = time()
	settings = ws.settings

	# create scaling variables
	# with scaling    -> uses mutable diagonal scaling matrices
	# without scaling -> uses identity matrices
	ws.sm = (settings.scaling > 0) ? ScaleMatrices(ws.p.m, ws.p.n) : ScaleMatrices()

	# perform preprocessing steps (scaling, initial KKT factorization)
	ws.times.setup_time = @elapsed setup!(ws);
	ws.times.proj_time  = 0. #reset projection time

	# instantiate variables
	num_iter = 0
	status = :Unsolved
	cost = Inf


	# print information about settings to the screen
	settings.verbose && print_header(ws)
	time_limit_start = time()

 	# iter_history = IterateHistory(ws.p.m ,ws.p.n)

	# update_iterate_history!(iter_history, ws.vars.x, ws.vars.s, -ws.vars.μ, ws.vars.v, ws.r_prim, ws.r_dual, zeros(10))

	#preallocate arrays
	m = ws.p.m
	n = ws.p.n
	δx = zeros(n)
	δy =  zeros(m)
	s_tl = zeros(m) # i.e. sTilde

	ls = zeros(n + m)
	sol = zeros(n + m)
	x_tl = view(sol, 1:n) # i.e. xTilde
	ν = view(sol, (n + 1):(n + m))
	v1 = view(ws.vars.v, 1:n)
	v2 = view(ws.vars.v, n+1:n+m)
	settings.verbose_timing && (iter_start = time())

	for iter = 1:settings.max_iter

		num_iter += 1

		if num_iter > 1 && num_iter < 300
			COSMO.update_history!(ws.accelerator, ws.vars.v, ws.vars.v_prev)
			COSMO.accelerate!(ws.vars.v, ws.vars.v_prev, ws.accelerator)
		end

		@. ws.vars.v_prev = ws.vars.v
		@. δx = ws.vars.x
		@. δy = ws.vars.μ


		ws.times.proj_time += admm_step!(
			ws.vars.x, ws.vars.s, ws.vars.μ, ws.vars.v, ν,
			x_tl, s_tl, ls, sol,
			ws.F, ws.p.q, ws.p.b, ws.ρvec,
			settings.alpha, settings.sigma,
			m, n, ws.p.C, v1, v2);
		# compute deltas for infeasibility detection
		@. δx = ws.vars.x - δx
		@. δy = -ws.vars.μ + δy

		# @show(ws.vars.v)
		# @show(ws.vars.s.data)
		if mod(iter, ws.settings.check_termination )  == 0
			calculate_residuals!(ws)
		end

		# eta = zeros(10)
		# if ws.settings.accelerator == :empty
		# 	eta = zeros(10)
		# else
		# 	eta = ws.accelerator.eta
		# 	ne = length(eta)
		# 	if ne < 10
		# 		eta = [eta; zeros(10-ne)]
		# 	end
		# end
		# update_iterate_history!(iter_history, ws.vars.x, ws.vars.s, -ws.vars.μ, ws.vars.v, ws.r_prim, ws.r_dual, eta)

		# check convergence with residuals every {settings.checkIteration} steps
		if mod(iter, settings.check_termination) == 0
			# update cost
			cost = ws.sm.cinv[] * (1/2 * ws.vars.x' * ws.p.P * ws.vars.x + ws.p.q' * ws.vars.x)[1]

			if abs(cost) > 1e20
				status = :Unsolved
				break
			end

			# print iteration steps
			settings.verbose && print_iteration(settings, iter, cost, ws.r_prim, ws.r_dual, ws.ρ)

			if has_converged(ws)
				status = :Solved
				break
			end
		end

		# check infeasibility conditions every {settings.checkInfeasibility} steps
		if mod(iter, settings.check_infeasibility) == 0
			if is_primal_infeasible(δy, ws)
				status = :Primal_infeasible
				cost = Inf
				break
			end

			if is_dual_infeasible(δx, ws)
				status = :Dual_infeasible
				cost = -Inf
				break
			end
		end



		if settings.time_limit !=0 &&  (time() - time_limit_start) > settings.time_limit
			status = :Time_limit_reached
			break
		end


		# adapt rhoVec if enabled
		if ws.settings.adaptive_rho && (mod(iter, ws.settings.adaptive_rho_interval + 1) == 0) && (ws.settings.adaptive_rho_interval + 1 > 0)
			adapt_rho_vec!(ws, iter)
		end

	end #END-ADMM-MAIN-LOOP

	settings.verbose_timing && (ws.times.iter_time = (time() - iter_start))
	settings.verbose_timing && (ws.times.post_time = time())

	# calculate primal and dual residuals
	if num_iter == settings.max_iter
		calculate_residuals!(ws)
		status = :Max_iter_reached
	end

	# reverse scaling for scaled feasible cases
	if settings.scaling != 0
		reverse_scaling!(ws)
		# FIXME: Another cost calculation is not necessary since cost value is not affected by scaling
		cost =  (1/2 * ws.vars.x' * ws.p.P * ws.vars.x + ws.p.q' * ws.vars.x)[1] #sm.cinv * not necessary anymore since reverseScaling
	end

	ws.times.solver_time = time() - solver_time_start
	settings.verbose_timing && (ws.times.post_time = time() - ws.times.post_time)
	# print solution to screen
	settings.verbose && print_result(status, num_iter, cost, ws.times.solver_time)

	# create result object
	res_info = ResultInfo(ws.r_prim, ws.r_dual)
	y = -ws.vars.μ

	aa_fail = 0
	if typeof(ws.accelerator) <: AndersonAccelerator{Float64}
		aa_fail = ws.accelerator.fail_counter
	end

	return Result{Float64}(ws.vars.x, y, ws.vars.s.data, cost, num_iter, status, res_info, ws.times), [ws.r_prim;ws.r_dual], 0

end

