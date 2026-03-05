"""
    struct EmbeddedCorrectionData

Stores precomputed data needed for embedded correction in non-admissible H-matrix blocks.
"""
struct EmbeddedCorrectionData{K,Q,V}
    quads_dict::Dict
    X::Vector  # target quadrature nodes
    Y::Q       # source quadrature
    kernel::K
    sing_order::V  # Val{N} for singularity order
end

"""
    create_correction_callback(iop::IntegralOperator, maxdist, quads_dict)

Create a callback function for embedded correction in non-admissible H-matrix blocks.
This callback will be passed to HMatrices.jl during assembly.
"""
function create_correction_callback(iop::IntegralOperator, quads_dict)
    # Precompute all necessary data
    X = collect(target(iop))  # target quadrature nodes
    Y = source(iop)           # source quadrature
    K = kernel(iop)           # kernel function
    
    # Compute singularity order
    msh = mesh(Y)
    geo_dim = geometric_dimension(msh)
    p = singularity_order(K)
    sing_order = if isnothing(p)
        @warn "missing method `singularity_order` for kernel. Assuming finite part integral."
        Val(-2)
    else
        Val(p + (geo_dim - 1))
    end
    
    # Create the data structure
    data = EmbeddedCorrectionData(
        quads_dict,
        X,
        Y,
        K,
        sing_order
    )
    
    # Return the callback function (closure)
    return function correction_callback(out, K_unused, irange, jrange, hmat)
        _fill_block_with_embedded_correction!(out, data, irange, jrange)
    end
end

"""
    _fill_block_with_embedded_correction!(out, data, irange, jrange)

Fill a dense matrix block with corrected values using adaptive quadrature.
This is called for non-admissible blocks where all interactions require correction.

Strategy: We compute the correction for all (i,j) pairs in the block by reusing
the logic from _adaptive_correction_etype!, but instead of building a sparse matrix,
we directly fill the dense block.
"""
function _fill_block_with_embedded_correction!(
    out::Matrix,
    data::EmbeddedCorrectionData,
    irange::UnitRange,
    jrange::UnitRange
)
    X = data.X
    Y = data.Y
    K = data.kernel
    msh = mesh(Y)
    
    Xqnodes = X
    Yqnodes = collect(Y)
    
    # Process by element type
    for E in element_types(msh)
        els = elements(msh, E)
        ori = orientation(msh, E)
        
        quads = merge(data.quads_dict[E], (regular_quad = quadrature_rule(Y, E),))
        L = lagrange_basis(quads.regular_quad)
        x̂ = qcoords(quads.regular_quad) |> collect
        el2qtags = etype2qtags(Y, E)
        
        elements_in_block = Int[]
        for n in 1:size(el2qtags, 2)
            jglob = view(el2qtags, :, n)
            if any(j -> j ∈ jrange, jglob)
                push!(elements_in_block, n)
            end
        end
        for n in elements_in_block
            el = els[n]
            ori_n = ori[n]
            jglob = view(el2qtags, :, n)
            
            # For each target point in irange
            for (iloc, iglob) in enumerate(irange)
                xnode = Xqnodes[iglob]
                
                # Find closest quadrature node
                dmin, j_closest = findmin(
                    k -> norm(coords(xnode) - coords(Yqnodes[jglob[k]])),
                    1:length(jglob),
                )
                x̂nearest = x̂[j_closest]
                
                # Compute corrected integral
                if iszero(dmin)
                    W = guiggiani_singular_integral(
                        K, L, x̂nearest, el, ori_n,
                        quads.radial_quad, quads.angular_quad,
                        data.sing_order,
                    )
                else
                    integrand = (ŷ) -> begin
                        y = el(ŷ)
                        jac = jacobian(el, ŷ)
                        ν = _normal(jac, ori_n)
                        τ′ = _integration_measure(jac)
                        M = K(xnode, (coords = y, normal = ν))
                        v = L(ŷ)
                        map(v -> M * v, v) * τ′
                    end
                    W = quads.nearfield_quad(integrand)
                end
                
                # Fill output for j in jrange
                for (k, jglob_k) in enumerate(jglob)
                    if jglob_k ∈ jrange
                        jloc = findfirst(==(jglob_k), jrange)
                        out[iloc, jloc] = W[k]
                    end
                end
            end
        end
    end
    
    return out
end
