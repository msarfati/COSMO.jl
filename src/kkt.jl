function factor_KKT!(ws::COSMO.Workspace)
	settings = ws.settings
	p = ws.p

	if length(ws.M) == 0
		# KKT matrix
		ws.M = [p.P + settings.sigma*I SparseMatrixCSC(p.A'); p.A -I]
	end
	n = p.n
	m = p.m
	@inbounds @simd for i = (n + 1):(n + m)
		ws.M[i, i] = -1.0 / ws.ρvec[i - n]
	end

	# Do LDLT Factorization: A = LDL^T
	if settings.verbose_timing
		ws.times.factor_time += @elapsed ws.F = ldlt(ws.M)
	else
		ws.F = ldlt(ws.M)
	end
	return nothing
end
