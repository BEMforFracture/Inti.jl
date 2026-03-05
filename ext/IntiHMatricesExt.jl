module IntiHMatricesExt

import Inti
import HMatrices

function __init__()
    return @info "Loading Inti.jl HMatrices extension"
end

# HMatrices interface
function Inti.assemble_hmatrix(
        iop::Inti.IntegralOperator;
        atol = 0,
        rank = typemax(Int),
        rtol = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64)),
        eta = 3,
        embedded_correction = false
    )
    comp = HMatrices.PartialACA(; rtol, atol, rank)
    adm = HMatrices.StrongAdmissibilityStd(eta)
    # Prepare correction callback if requested
    correction_callback = if embedded_correction
        # Compute correction parameters
        if isnothing(maxdist) || isnothing(correction_rtol) || isnothing(correction_atol)
            maxdist_, rtol_, atol_ = Inti.local_correction_dist_and_tol(iop)
            maxdist = isnothing(maxdist) ? maxdist_ : maxdist
            correction_rtol = isnothing(correction_rtol) ? rtol_ : correction_rtol
            correction_atol = isnothing(correction_atol) ? atol_ : correction_atol
        end
        
        # Build quadrature dictionary
        msh = Inti.mesh(Inti.source(iop))
        quads_dict = Dict()
        for E in Inti.element_types(msh)
            ref_domain = Inti.reference_domain(E)
            quads = (
                nearfield_quad = Inti.adaptive_quadrature(ref_domain; rtol=correction_rtol, atol=correction_atol),
                radial_quad = Inti.adaptive_quadrature(Inti.ReferenceLine(); rtol=correction_rtol, atol=correction_atol),
                angular_quad = Inti.adaptive_quadrature(Inti.ReferenceLine(); rtol=correction_rtol, atol=correction_atol),
            )
            quads_dict[E] = quads
        end
        
        # Create the callback
        Inti.create_correction_callback(iop, maxdist, quads_dict)
    else
        nothing
    end
    X = [Inti.coords(x) for x in iop.target]
    Y = iop.target === iop.source ? X : [Inti.coords(y) for y in iop.source]
    Xclt = HMatrices.ClusterTree(X; copy_elements = false)
    Yclt = X === Y ? Xclt : HMatrices.ClusterTree(Y; copy_elements = false)
    return HMatrices.assemble_hmatrix(
        iop, Xclt, Yclt; 
        adm, comp, 
        correction_callback,
        global_index = true
    )
end

end # module
