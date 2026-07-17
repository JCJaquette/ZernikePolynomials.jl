# =====================================================================================
# export_MMT_for_matlab.jl
#
# Run-once exporter that produces the data of MATLAB's construct_MMT_data(N, aa=1)
# --- rigorously, in multiple-precision interval arithmetic --- and writes it to
# .mat files that the MATLAB side can load as INTLAB intervals OR as doubles.
#
# Companion to: Cadiot, Jaquette, Lessard, Takayasu, "Validated matrix multiplication
# transform for orthogonal polynomials ...", CNSNS 2025.
#
# Reuses the public functions of list_functions.jl (eig_multiplication, proof_full_eigen,
# computation_W). The only genuinely new algorithm here is japolym_interval (the Jacobi
# three-term / Forsythe recurrence at arbitrary nodes). No edits to list_functions.jl.
#
# Usage:
#   julia export_MMT_for_matlab.jl quick          # N = 10, 20 (debug, ~minutes)
#   julia export_MMT_for_matlab.jl production      # N = 10:10:150 (overnight)
#   julia -t 14 export_MMT_for_matlab.jl production paranoid   # (parallel added later)
# =====================================================================================

using RadiiPolynomial, IntervalArithmetic, LinearAlgebra, MAT, Printf, Dates, Distributed

const _HERE = @__DIR__
include(joinpath(_HERE, "list_functions.jl"))

# Input bases for the iMMT_data columns, matching SpectralCoeff.iMMTcolumn:
#   column i -> (k_in, m_in);  (0,1)->1, (1,0)->2, (1,2)->3
const BASES = [(0, 1), (1, 0), (1, 2)]

# -------------------------------------------------------------------------------------
# Parallelism lives in the Distributed path below (eigen_one / master_kout_dist /
# export_all_dist): one validated eigenpair per task, across worker PROCESSES so each has
# its own GC heap. The serial list_functions.jl kernel proof_full_eigen is therefore reused
# unmodified (correctness gate + quick mode) and there is no threaded copy to keep in sync.
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# japolym_interval(deg, k, m, nodes)
#
# Jacobi polynomials P_j^{k,m} (paper Def 2.1; MATLAB japolym convention) evaluated at
# `nodes`, via the three-term (Forsythe) recurrence in interval BigFloat.
# Returns an (length(nodes) x (deg+1)) matrix with entry [l, j] = P_{j-1}^{k,m}(nodes[l]).
# This is MATLAB `japolym(deg, k, m, nodes)'`.
# -------------------------------------------------------------------------------------
function japolym_interval(deg, k, m, nodes::AbstractVector{T}) where {T}
    L = length(nodes)
    apb = k + m
    Y = Matrix{T}(undef, L, deg + 1)
    @views Y[:, 1] .= interval(big(1.0))                         # P_0 = 1
    if deg >= 1
        half = interval(big(1)) / interval(big(2))
        @views Y[:, 2] .= half .* (interval(big(k - m)) .+ interval(big(apb + 2)) .* nodes)
    end
    for j in 2:deg
        a1 = interval(big(2) * big(j) * big(j + apb) * big(2 * j + apb - 2))
        a2 = interval(big(2 * j + apb - 1) * (big(k)^2 - big(m)^2))
        b3 = big(2 * j + apb - 2)
        a3 = interval(b3 * (b3 + 1) * (b3 + 2))
        a4 = interval(big(2) * big(j + k - 1) * big(j + m - 1) * big(2 * j + apb))
        @views Y[:, j + 1] .= ((a2 .+ a3 .* nodes) .* Y[:, j] .- a4 .* Y[:, j - 1]) ./ a1
    end
    return Y
end

# -------------------------------------------------------------------------------------
# assemble_kout: from the validated eigenvectors (columns) and eigenvalues of the (k_out,1)
# Jacobi tridiagonal, build nodes, nodes_Z, Gauss weights omega, inner-product weights W,
# and Grid2Spec (= MATLAB MMT_data{k_out+1,1} = paper iMMT). Shared by the serial and the
# Distributed paths, so the assembly math lives in exactly one place.
# -------------------------------------------------------------------------------------
function assemble_kout(eigenvectors, eigenvalues, n, k_out)
    N = n - 1
    nodes = eigenvalues                                          # length n, ascending
    nodes_Z = sqrt.((nodes .+ interval(big(1))) ./ interval(big(2)))
    μ0 = interval(big(2))^(k_out + 2) * interval(factorial(big(k_out))) *
         interval(factorial(big(1))) / interval(factorial(big(k_out + 2)))
    ω = μ0 .* (eigenvectors[1, :] .^ 2)                          # Gauss weights (paper eq 16)
    W = computation_W(k_out, 1, N, 0)                            # inner-product weights γ_n
    J = japolym_interval(N, k_out, 1, nodes)                     # (n x n) = paper MMT
    Grid2Spec = (interval(big(1)) ./ W) .* (J') .* (ω')          # (n x n) = MMT_data{k+1,1}
    return (nodes=nodes, nodes_Z=nodes_Z, Grid2Spec=Grid2Spec, omega=ω, W=W)
end

# -------------------------------------------------------------------------------------
# master_kout — serial path (correctness gate + quick mode). Runs the reused O(n^4) kernel
# proof_full_eigen in-process, then assemble_kout. (Production uses master_kout_dist.)
# -------------------------------------------------------------------------------------
function master_kout(n, k_out, precision, eps_N; eigfun=proof_full_eigen)
    setprecision(precision)
    N = n - 1
    pspace = ParameterSpace() × ParameterSpace()^(N + 1)
    Bf = mid.(eig_multiplication(N, k_out, 1, 1))                 # Float64 tridiagonal
    E = eigen(Hermitian(Matrix(Bf)))
    Bbig = Matrix(eig_multiplication(N, interval(big(k_out)), interval(big(1)), 0))
    eigenvectors, eigenvalues = eigfun(E, Bbig, N, pspace, eps_N) # <-- reused O(n^4) kernel
    return assemble_kout(eigenvectors, eigenvalues, n, k_out)
end

# -------------------------------------------------------------------------------------
# build_N_from(outs, N): assemble the 2x3 MMT_data / iMMT_data cells (construct_MMT_data(N,1)
# layout, n = 2N-1) from the per-basis outputs `outs = [out_k0, out_k1]`. Cross-basis blocks
# evaluate one basis's Jacobi polynomials at another basis's validated nodes (Forsythe).
# -------------------------------------------------------------------------------------
function build_N_from(outs, N)
    MMT_data = Matrix{Any}(undef, 2, 3)
    iMMT_data = Matrix{Any}(undef, 2, 3)
    for kk in 0:1
        r = outs[kk + 1]
        MMT_data[kk + 1, 1] = r.Grid2Spec
        MMT_data[kk + 1, 2] = r.nodes
        MMT_data[kk + 1, 3] = r.nodes_Z
        for (i, (k_in, m_in)) in enumerate(BASES)
            Jc = japolym_interval(N - 1, k_in, m_in, r.nodes)    # (n x N)
            if m_in == 2
                Jc = (r.nodes_Z .^ 2) .* Jc
            end
            iMMT_data[kk + 1, i] = Jc
        end
    end
    return MMT_data, iMMT_data
end

# Serial build (correctness gate + quick mode): n = 2N-1.
function build_N(N; precision=256, eps_N=1e-40, eigfun=proof_full_eigen)
    n = 2N - 1
    outs = [master_kout(n, kk, precision, eps_N; eigfun=eigfun) for kk in 0:1]
    return build_N_from(outs, N)
end

# -------------------------------------------------------------------------------------
# Convert an Interval{BigFloat} array to the tightest double enclosure (directed rounding),
# and report the widest entry in ULPs. Vectors are returned as n x 1 columns (MATLAB shape).
# -------------------------------------------------------------------------------------
function to_double_infsup(A)
    lo = Float64.(inf.(A), RoundDown)
    hi = Float64.(sup.(A), RoundUp)
    # Matrix scale, and a floor so that near-zero entries (whose true value is ~0 and whose
    # enclosure is absolutely tiny) do not blow the relative ULP up: an entry's width is
    # measured against its own magnitude OR against ~1 ULP at the matrix scale, whichever is
    # larger. maxulp <= 1 then means every entry is enclosed to within one rounding.
    scale = 0.0
    @inbounds for i in eachindex(lo)
        scale = max(scale, abs(lo[i]), abs(hi[i]))
    end
    floorv = scale == 0.0 ? floatmin(Float64) : scale * eps(1.0)
    maxulp = 0.0
    @inbounds for i in eachindex(lo)
        w = hi[i] - lo[i]
        w == 0.0 && continue
        denom = max(eps(max(abs(lo[i]), abs(hi[i]))), floorv)
        maxulp = max(maxulp, w / denom)
    end
    if ndims(lo) == 1
        lo = reshape(lo, :, 1); hi = reshape(hi, :, 1)
    end
    return lo, hi, maxulp
end

# Pack one N into a MAT-writable Dict; also return the worst-case ULP width over all blocks.
function pack(N, MMT_data, iMMT_data, precision, eps_N)
    n = 2N - 1
    d = Dict{String,Any}(
        "N" => Float64(N), "aa" => 1.0, "n" => Float64(n),
        "precision" => Float64(precision), "eps_N" => Float64(eps_N),
        "format_version" => "1.0", "created" => string(Dates.now()),
        "bases" => Float64[0 1; 1 0; 1 2],                      # (k_in,m_in) per iMMT column
    )
    maxulp = 0.0
    for kk in 0:1
        for (name, A) in (("Grid2Spec", MMT_data[kk + 1, 1]),
                          ("nodes", MMT_data[kk + 1, 2]),
                          ("nodesZ", MMT_data[kk + 1, 3]))
            lo, hi, mu = to_double_infsup(A)
            d["$(name)_k$(kk)_lo"] = lo
            d["$(name)_k$(kk)_hi"] = hi
            maxulp = max(maxulp, mu)
        end
        for i in 1:3
            lo, hi, mu = to_double_infsup(iMMT_data[kk + 1, i])
            d["iMMT_k$(kk)_c$(i)_lo"] = lo
            d["iMMT_k$(kk)_c$(i)_hi"] = hi
            maxulp = max(maxulp, mu)
        end
    end
    d["maxULP"] = maxulp
    return d, maxulp
end

# -------------------------------------------------------------------------------------
# export_N: build, verify <=1 ULP, bump precision + rebuild if not, then matwrite one file.
# -------------------------------------------------------------------------------------
# eps_N=nothing (default) derives eps_N from precision via eps_N_for and RECOMPUTES it on
# each precision bump; pass a number to pin a fixed eps_N (debugging / correctness gate).
function export_N(N; precision, eps_N=nothing, outdir, eigfun=proof_full_eigen,
                  max_bumps=3, ulp_tol=1.0 + 1e-6)
    p = precision
    local d, maxulp
    ok = false
    for attempt in 0:max_bumps
        en = eps_N === nothing ? eps_N_for(p) : eps_N
        t = @elapsed begin
            MMT_data, iMMT_data = build_N(N; precision=p, eps_N=en, eigfun=eigfun)
            d, maxulp = pack(N, MMT_data, iMMT_data, p, en)
        end
        @printf("  N=%d n=%d prec=%d eps_N=%.2g  maxULP=%.3g  (%.1fs)\n", N, 2N - 1, p, en, maxulp, t)
        flush(stdout)
        if maxulp <= ulp_tol
            ok = true
            break
        end
        p += 128                                                # not tight enough -> more bits (eps_N tightens with it)
    end
    mkpath(outdir)
    fname = joinpath(outdir, @sprintf("MMT_data_N%03d.mat", N))
    matwrite(fname, d)
    return (N=N, n=2N - 1, precision=p, maxULP=maxulp, ok=ok, file=fname)
end

# -------------------------------------------------------------------------------------
# Driver.
# -------------------------------------------------------------------------------------
default_precision(n) = max(256, 64 * cld(round(Int, 2.5 * n), 64))

# eps_N policy — the Newton residual target for newton_eig, as a function of working
# precision. newton_eig drives the eigenpair residual ‖B·V − v·V‖ down to eps_N; that
# residual floor is ~2^-precision (reachable regardless of eigenvalue clustering). The
# VALIDATED node radius is rmin ≈ residual × ‖inv(T)‖, and ‖inv(T)‖ (the eigenvalue-gap
# conditioning) blows up with n — ~2^90 at n=151. A FIXED eps_N=1e-40 therefore pinned
# rmin at 1e-40·2^90 ≈ 1e-13 ≈ 480 ULP at n=151, and bumping precision alone never moved
# it (Newton still stopped at the 1e-40 target). Tying eps_N to precision at 2^(-0.9·p)
# keeps it ~10% above the 2^-p residual floor — reached in a few quadratic Newton steps —
# yet vastly tighter than 1e-40, so rmin tracks precision and each bump genuinely tightens
# it. Returned as Float64 (representable for all p ≤ ~1000 we use; underflow only past that).
eps_N_for(precision) = 2.0^(-0.9 * precision)

function export_all(; Nlist, outdir, precision_fn=default_precision, eps_N=nothing,
                    eigfun=proof_full_eigen)
    mkpath(outdir)
    println("Exporting to: ", outdir, "  (serial, eigfun=", nameof(eigfun), ")")
    results = NamedTuple[]
    for N in Nlist
        r = export_N(N; precision=precision_fn(2N - 1), eps_N=eps_N, outdir=outdir, eigfun=eigfun)
        push!(results, r)
        @printf("N=%-4d n=%-4d prec=%-4d maxULP=%.3g  %s  -> %s\n",
                r.N, r.n, r.precision, r.maxULP, r.ok ? "OK" : "!! >1 ULP", basename(r.file))
        flush(stdout)
    end
    println("\nSummary:")
    for r in results
        @printf("  N=%-4d maxULP=%.3g  %s\n", r.N, r.maxULP, r.ok ? "OK" : "!! NOT 1-ULP")
    end
    return results
end

# =====================================================================================
# Distributed (multiprocessing) path — Option B: one validated eigenpair per task.
# Worker PROCESSES (not threads), so each has its own GC heap and BigFloat allocation in
# one worker never stalls another (the shared-heap GC contention that capped threads ~2x).
# The serial kernel proof_full_eigen is reused verbatim for the correctness gate; the only
# per-eigenpair code duplicated from it is `eigen_one` below (smaller than a threaded copy).
# =====================================================================================

# eigen_one — one validated eigenpair, the unit of Distributed work. Receives only the small
# Float64 eigen starting guess (v, V); reconstructs the cheap interval tridiagonal locally
# (so O(n^2) BigFloat matrices are never shipped). Sole duplication of proof_full_eigen's
# loop body, kept minimal on purpose.
function eigen_one(N, k_out, m, vstart, Vstart, precision, eps_N)
    setprecision(precision)
    pspace = ParameterSpace() × ParameterSpace()^(N + 1)
    Bbig = Matrix(eig_multiplication(N, interval(big(k_out)), interval(big(1)), 0))
    U = Sequence(pspace, vec([big(vstart); big.(Vstart)]))
    U = newton_eig(mid.(Bbig), mid.(U), eps_N)
    U = interval.(U)
    P, rmin = proof_eigen(Bbig, U)
    P == 1 || error("eigen_one: eigenpair N=$N k=$k_out m=$m failed to validate")
    col = interval.(inf.(coefficients(component(U, 2))) .- rmin,
                    sup.(coefficients(component(U, 2))) .+ rmin)
    dval = interval(inf(component(U, 1)[1]) - rmin, sup(component(U, 1)[1]) + rmin)
    return (m=m, col=col, dval=dval)
end

# master_kout_dist — like master_kout, but the N+1 eigenpair proofs run as Distributed tasks
# over `pool`. E (Float64) is computed once on the master; each task gets one column's start.
function master_kout_dist(n, k_out, precision, eps_N, pool)
    setprecision(precision)
    N = n - 1
    Bf = mid.(eig_multiplication(N, k_out, 1, 1))
    E = eigen(Hermitian(Matrix(Bf)))
    starts = [(m, E.values[m], collect(E.vectors[:, m])) for m in 1:N+1]
    results = pmap(pool, starts) do s
        eigen_one(N, k_out, s[1], s[2], s[3], precision, eps_N)
    end
    M = interval.(big.(zeros(N + 1, N + 1)))
    d = interval.(big.(zeros(N + 1)))
    for r in results
        @views M[:, r.m] .= r.col
        d[r.m] = r.dval
    end
    return assemble_kout(M, d, n, k_out)
end

# export_N_dist — build one N via the Distributed eigenpair proofs, verify <=1 ULP (bump
# precision + rebuild if not), write one .mat. Same output/contract as export_N.
# eps_N=nothing (default) derives eps_N from precision (recomputed on each bump); see export_N.
function export_N_dist(N; precision, eps_N=nothing, outdir, pool,
                       max_bumps=3, ulp_tol=1.0 + 1e-6)
    p = precision
    local d, maxulp
    ok = false
    for attempt in 0:max_bumps
        en = eps_N === nothing ? eps_N_for(p) : eps_N
        t = @elapsed begin
            outs = [master_kout_dist(2N - 1, kk, p, en, pool) for kk in 0:1]
            MMT_data, iMMT_data = build_N_from(outs, N)
            d, maxulp = pack(N, MMT_data, iMMT_data, p, en)
        end
        @printf("  N=%d n=%d prec=%d eps_N=%.2g  maxULP=%.3g  (%.1fs)\n", N, 2N - 1, p, en, maxulp, t)
        flush(stdout)
        if maxulp <= ulp_tol
            ok = true
            break
        end
        p += 128
    end
    mkpath(outdir)
    fname = joinpath(outdir, @sprintf("MMT_data_N%03d.mat", N))
    matwrite(fname, d)
    return (N=N, n=2N - 1, precision=p, maxULP=maxulp, ok=ok, file=fname)
end

# machine_report — one-time hardware/config block for the run log, so a past run's per-N
# timings can be read in the context of the machine that produced them (and reused to
# estimate future runs). Returns a printable string.
function machine_report(; nw, heap_hint)
    io = IOBuffer()
    cpus = Sys.cpu_info()
    cpumodel = isempty(cpus) ? "?" : strip(cpus[1].model)
    cpuspeed = isempty(cpus) ? 0 : cpus[1].speed
    blasstr = try
        join([basename(String(l.libname)) for l in BLAS.get_config().loaded_libs], ", ")
    catch
        "(unknown)"
    end
    println(io, "="^72)
    println(io, "MMT/iMMT validated export — run log")
    println(io, "started:          ", Dates.now())
    println(io, "-"^72)
    println(io, "CPU model:        ", cpumodel)
    @printf(io, "CPU base clock:   %d MHz\n", cpuspeed)
    println(io, "logical procs:    ", Sys.CPU_THREADS, "  (Sys.cpu_info entries: ", length(cpus), ")")
    @printf(io, "total RAM:        %.1f GB\n", Sys.total_memory() / 2^30)
    @printf(io, "free RAM (start): %.1f GB\n", Sys.free_memory() / 2^30)
    println(io, "machine / kernel: ", Sys.MACHINE, "  /  ", Sys.KERNEL, " ", Sys.WORD_SIZE, "-bit")
    println(io, "Julia version:    ", VERSION)
    println(io, "BLAS libs:        ", blasstr, "  (num_threads=1 per worker)")
    println(io, "worker procs:     ", nw, "   (+1 master)")
    println(io, "--heap-size-hint: ", heap_hint, " per worker")
    println(io, "precision sched:  default_precision(n) = max(256, 64*ceil(2.5n/64)) bits")
    println(io, "eps_N policy:     eps_N_for(p) = 2^(-0.9 p)")
    println(io, "="^72)
    return String(take!(io))
end

# export_all_dist — spin up `nw` worker processes (own heaps, capped by --heap-size-hint),
# load the code on each, then export every N biggest-first (fail fast on memory/precision).
# A run log (machine specs + per-N timing) is written incrementally to `logfile` (default:
# <repo>/logs/timing_<timestamp>.txt) so partial timings survive a crash and future long
# runs can be estimated from past hardware-stamped data.
function export_all_dist(; Nlist, outdir, precision_fn=default_precision, eps_N=nothing,
                         nw::Int=8, heap_hint::AbstractString="1500M", logfile=nothing)
    mkpath(outdir)
    if logfile === nothing
        logdir = joinpath(_HERE, "logs")
        mkpath(logdir)
        logfile = joinpath(logdir, "timing_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * ".txt")
    else
        mkpath(dirname(logfile))
    end
    thisfile = joinpath(_HERE, "export_MMT_for_matlab.jl")
    need = (nw + 1) - nprocs()
    need > 0 && addprocs(need; exeflags=["--heap-size-hint=$heap_hint"])
    for pid in workers()
        remotecall_wait(Base.include, pid, Main, thisfile)  # defines eigen_one etc. (CLI guarded)
        remotecall_wait(BLAS.set_num_threads, pid, 1)       # no BLAS oversubscription per worker
    end
    pool = WorkerPool(workers())

    header = machine_report(; nw=nworkers(), heap_hint=heap_hint)
    colhdr = @sprintf("%-5s %-5s %-6s %-10s %-9s %-9s %s",
                      "N", "n", "prec", "eps_N", "maxULP", "wall_s", "status")
    open(logfile, "w") do io
        print(io, header)
        println(io, colhdr)
    end
    print(header)
    println("Exporting to: ", outdir)
    println("Log file:     ", logfile)
    println(colhdr); flush(stdout)

    results = NamedTuple[]
    run_t0 = time()
    for N in sort(collect(Nlist); rev=true)                 # biggest N first
        el = @elapsed r = export_N_dist(N; precision=precision_fn(2N - 1), eps_N=eps_N,
                                        outdir=outdir, pool=pool)
        push!(results, r)
        en = eps_N === nothing ? eps_N_for(r.precision) : eps_N
        line = @sprintf("%-5d %-5d %-6d %-10.2e %-9.4g %-9.1f %s",
                        r.N, r.n, r.precision, en, r.maxULP, el, r.ok ? "OK" : "!! >1ULP")
        open(logfile, "a") do io
            println(io, line)
            flush(io)
        end
        println(line); flush(stdout)
    end
    total = time() - run_t0

    sort!(results; by=r -> r.N)
    open(logfile, "a") do io
        println(io, "-"^72)
        @printf(io, "total wall:       %.1f s  (%.2f h)  over %d values of N\n",
                total, total / 3600, length(results))
        @printf(io, "free RAM (end):   %.1f GB\n", Sys.free_memory() / 2^30)
        println(io, "finished:         ", Dates.now())
    end
    println("\nSummary:")
    for r in results
        @printf("  N=%-4d maxULP=%.3g  %s\n", r.N, r.maxULP, r.ok ? "OK" : "!! NOT 1-ULP")
    end
    @printf("Total wall: %.1f s (%.2f h).  Log: %s\n", total, total / 3600, logfile)
    return results
end

# ------------------------------- command-line entry ----------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    mode = length(ARGS) >= 1 ? ARGS[1] : "quick"
    outdir = joinpath(_HERE, "MMT_export_data")
    if mode == "quick"                        # serial, small: end-to-end smoke test
        export_all(Nlist=[10, 20], outdir=outdir, precision_fn=(n -> 256))
    elseif mode == "distquick"                # Distributed, small: exercise the worker path
        nw = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 3
        export_all_dist(Nlist=[10, 20], outdir=outdir, precision_fn=(n -> 256), nw=nw)
    elseif mode == "production"               # Distributed sweep: production [nw] [Nmax]
        nw = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
        Nmax = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 150
        export_all_dist(Nlist=10:10:Nmax, outdir=outdir, nw=nw)
    else
        error("unknown mode $mode (use \"quick\" | \"distquick [nw]\" | \"production [nw] [Nmax]\")")
    end
end
