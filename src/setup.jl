function setup!(ws::COSMO.Workspace)
	# scale problem data
	if ws.settings.scaling != 0
		scale_ruiz!(ws)
	end

	set_rho_vec!(ws)

	# factor the KKT condition matrix
	ws.flags.FACTOR_LHS && factor_KKT!(ws)

  # instantiate accelerator
  set_accelerator!(ws)
end

function set_accelerator!(ws::COSMO.Workspace)
  # if the user passed in a custom AbstractAccelerator, e.g. with different value for memory, don't change it
  if typeof(ws.accelerator) == EmptyAccelerator{Float64}
    if ws.settings.accelerator == :empty
      nothing
    elseif ws.settings.accelerator == :anderson1
      ws.accelerator = AndersonAccelerator{Float64}(2 * ws.p.m + ws.p.n, is_type1 = true)
    elseif ws.settings.accelerator == :anderson2
      ws.accelerator = AndersonAccelerator{Float64}(2 * ws.p.m + ws.p.n, is_type1 = false)
    else
      @warn("Your specification for settings.accelerator = $(ws.settings.accelerator) is unknown. Continue without acceleration.")
      ws.accelerator = EmptyAccelerator{Float64}()
    end
  end

end
