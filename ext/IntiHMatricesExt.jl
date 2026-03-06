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
        _, rtol_corr, atol_corr = Inti.local_correction_dist_and_tol(iop)
        
        # Build quadrature dictionary
        msh = Inti.mesh(Inti.source(iop))
        quads_dict = Dict()
        for E in Inti.element_types(msh)
            ref_domain = Inti.reference_domain(E)
            quads = (
                nearfield_quad = Inti.adaptive_quadrature(ref_domain; rtol=rtol_corr, atol=atol_corr),
                radial_quad = Inti.adaptive_quadrature(Inti.ReferenceLine(); rtol=rtol_corr, atol=atol_corr),
                angular_quad = Inti.adaptive_quadrature(Inti.ReferenceLine(); rtol=rtol_corr, atol=atol_corr),
            )
            quads_dict[E] = quads
        end
        
        # Create the callback
        Inti.create_correction_callback(iop, quads_dict)
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
