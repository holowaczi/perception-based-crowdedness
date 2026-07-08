using CSV, DataFrames, Statistics, LinearAlgebra, Clustering, Glob

# ─────────────────────────────────────────────────────────────────────────────
# Configuration – density methods to compute
# FOV tag format: "fov_<total_fov_degrees>_c<c×10>"
#   fov_210_c0  →  210° total FOV,  c=0.0 (outside FOV ignored)
#   fov_210_c5  →  210° total FOV,  c=0.5
# ─────────────────────────────────────────────────────────────────────────────
const METHODS = [
    "all",
    "fov_210_c0",
    "fov_210_c5",
]

function group_by_displacement(pos, n_agents, n_frames, α)
    disp_angles = Float64[]
    agent_ids   = Int[]
    for i in 1:n_agents
        first_f = findfirst(t -> all(isfinite.(pos[t, i, :])), 1:n_frames)
        last_f  = findlast( t -> all(isfinite.(pos[t, i, :])), 1:n_frames)
        (first_f === nothing || last_f === nothing || first_f == last_f) && continue
        dx = pos[last_f, i, 1] - pos[first_f, i, 1]
        dy = pos[last_f, i, 2] - pos[first_f, i, 2]
        push!(disp_angles, atan(dy, dx))
        push!(agent_ids, i)
    end
    isempty(agent_ids) && return collect(1:n_agents), Int[]
    α == 0 && return agent_ids, Int[]
    N = length(disp_angles)
    N < 2 && return agent_ids, Int[]
    unit_vecs = vcat(cos.(disp_angles)', sin.(disp_angles)')
    km   = kmeans(unit_vecs, 2; maxiter=200)
    asgn = assignments(km)
    gA = [agent_ids[k] for k in 1:N if asgn[k] == 1]
    gB = [agent_ids[k] for k in 1:N if asgn[k] == 2]
    return gA, gB
end

function density_agent(pos_t, i, nbrs, epsilon)
    all(isfinite.(pos_t[i, :])) || return NaN
    s = 0.0
    for j in nbrs
        j == i && continue
        all(isfinite.(pos_t[j, :])) || continue
        s += 1.0 / (sum((pos_t[i, :] .- pos_t[j, :]).^2) + epsilon^2)
    end
    return s
end

# Generalised FOV density
#   fov_half_deg : half-angle of FOV in degrees  (total FOV = 2 × fov_half_deg)
#   c            : weight for neighbours OUTSIDE FOV  (0=ignore, 1=full)
#
#   ρ_i = Σ_{j inside FOV}  1/(d²+ε²)  +  c · Σ_{j outside FOV}  1/(d²+ε²)
function density_agent_fov(pos_t, pos_prev, i, nbrs, epsilon, fov_half_deg, c)
    all(isfinite.(pos_t[i, :]))    || return NaN
    all(isfinite.(pos_prev[i, :])) || return NaN
    dir      = pos_t[i, :] .- pos_prev[i, :]
    dir_norm = norm(dir)
    # stationary agent: fall back to full density
    dir_norm < 1e-9 && return density_agent(pos_t, i, nbrs, epsilon)
    dir_unit  = dir ./ dir_norm
    cos_limit = cos(deg2rad(fov_half_deg))
    s = 0.0
    for j in nbrs
        j == i && continue
        all(isfinite.(pos_t[j, :])) || continue
        to_j      = pos_t[j, :] .- pos_t[i, :]
        to_j_norm = norm(to_j)
        to_j_norm < 1e-9 && continue
        contrib = 1.0 / (to_j_norm^2 + epsilon^2)
        if dot(dir_unit, to_j ./ to_j_norm) >= cos_limit
            s += contrib
        else
            s += c * contrib
        end
    end
    return s
end

function parse_fov_tag(tag)
    m = match(r"^fov_(\d+)_c(\d+)$", tag)
    m === nothing && return nothing
    total_deg = parse(Int, m.captures[1])
    c_int     = parse(Int, m.captures[2])
    return (half_deg = total_deg / 2.0, c = c_int / 10.0)
end

function compute_vrho(input_folder="trajectories",
                      file_detail_path="file_detail.csv")
    fps     = 120
    epsilon = 0.1

    detail_df = CSV.read(file_detail_path, DataFrame)

    function parse_ti_tf(raw)
        ismissing(raw) && return -1
        s = strip(string(raw))
        (s == "xxx" || s == "" || s == "missing") && return -1
        v = tryparse(Int, s)
        return v === nothing ? -1 : v
    end

    file_info = Dict{Int, NamedTuple{(:angle,:Ti,:Tf),Tuple{Int,Int,Int}}}()
    for row in eachrow(detail_df)
        file_info[Int(row[1])] = (
            angle = Int(row[2]),
            Ti    = parse_ti_tf(row[5]),
            Tf    = parse_ti_tf(row[6])
        )
    end

    script_dir = @__DIR__
    base_dir   = joinpath(script_dir, "csv_data")

    files = glob("bf_CF*.csv", input_folder)

    for file in files
        fname  = basename(file)
        m      = match(r"bf_CF(\d+)", fname)
        m === nothing && (println("Skipping $fname – cannot parse index"); continue)
        findex = parse(Int, m.captures[1])
        !haskey(file_info, findex) && (println("Skipping $fname – not in file_detail.csv"); continue)

        info = file_info[findex]
        α    = info.angle

        df       = CSV.read(file, DataFrame)
        n_agents = Int(floor((ncol(df) - 1) / 4))
        n_frames = nrow(df)

        pos = fill(NaN, n_frames, n_agents, 2)
        for i in 1:n_agents
            raw_x = df[:, 4*i-1]
            raw_y = df[:, 4*i+1]
            for t in 1:n_frames
                pos[t, i, 1] = ismissing(raw_x[t]) ? NaN : Float64(raw_x[t])
                pos[t, i, 2] = ismissing(raw_y[t]) ? NaN : Float64(raw_y[t])
            end
        end
        finite_vals = filter(isfinite, vec(pos))
        !isempty(finite_vals) && median(abs.(finite_vals)) > 50.0 && (pos .*= 0.001)

        gA, gB     = group_by_displacement(pos, n_agents, n_frames, α)
        all_agents = vcat(gA, gB)
        gA_set     = Set(gA)

        if info.Ti == -1 || info.Tf == -1
            Ti = 480; Tf = n_frames - 480
        else
            Ti = max(2, info.Ti); Tf = min(info.Tf, n_frames)
        end
        Tf <= Ti && (println("Skipping $fname – Tf ≤ Ti"); continue)

        # ── Expected direction per group (mean total displacement, unit vector) ──
        function group_expected_dir(grp)
            dx_sum, dy_sum, n = 0.0, 0.0, 0
            for i in grp
                first_f = findfirst(t -> all(isfinite.(pos[t, i, :])), 1:n_frames)
                last_f  = findlast( t -> all(isfinite.(pos[t, i, :])), 1:n_frames)
                (first_f === nothing || last_f === nothing || first_f == last_f) && continue
                dx_sum += pos[last_f, i, 1] - pos[first_f, i, 1]
                dy_sum += pos[last_f, i, 2] - pos[first_f, i, 2]
                n += 1
            end
            n == 0 && return [NaN, NaN]
            mag = sqrt(dx_sum^2 + dy_sum^2)
            mag < 1e-9 && return [NaN, NaN]
            return [dx_sum / mag, dy_sum / mag]
        end

        exp_dir_A = group_expected_dir(gA)
        exp_dir_B = isempty(gB) ? [NaN, NaN] : group_expected_dir(gB)

        n_rows      = Tf - Ti
        t_prime_col = Vector{Float64}(undef, n_rows)
        V           = fill(NaN, n_rows, n_agents)
        ACC         = fill(NaN, n_rows, n_agents)   # forward-difference acceleration [m/s²]
        DANGLE      = fill(NaN, n_rows, n_agents)
        DELTA       = fill(NaN, n_rows, n_agents)   # deviation from expected dir [deg], signed
        RHO         = Dict(method => fill(NaN, n_rows, n_agents) for method in METHODS)

        W = 60   # 60 frames = 0.5s window for direction estimation

        row_idx = 0
        for t in (Ti+1):Tf
            row_idx += 1
            t_prime_col[row_idx] = (t - Ti) / (Tf - Ti)
            pos_t    = pos[t,   :, :]
            pos_prev = pos[t-1, :, :]

            for i in all_agents
                # ── Velocity ──────────────────────────────────────────────────
                if all(isfinite.(pos[t, i, :])) && all(isfinite.(pos[t-1, i, :]))
                    V[row_idx, i] = norm(pos[t, i, :] .- pos[t-1, i, :]) * fps
                end

                # ── Direction change [degrees] ────────────────────────────────
                # d1 = displacement over [t-2W, t-W], d2 = displacement over [t-W, t]
                # Using window W smooths out frame-to-frame noise at 120fps
                t0, t1, t2 = t - 2*W, t - W, t
                if t0 >= 1 &&
                   all(isfinite.(pos[t2, i, :])) &&
                   all(isfinite.(pos[t1, i, :])) &&
                   all(isfinite.(pos[t0, i, :]))
                    d1 = pos[t1, i, :] .- pos[t0, i, :]
                    d2 = pos[t2, i, :] .- pos[t1, i, :]
                    n1, n2 = norm(d1), norm(d2)
                    if n1 > 1e-9 && n2 > 1e-9
                        u1 = d1 ./ n1
                        u2 = d2 ./ n2
                        sinθ = u1[1]*u2[2] - u1[2]*u2[1]   # 2-D cross product
                        cosθ = clamp(dot(u1, u2), -1.0, 1.0)
                        DANGLE[row_idx, i] = rad2deg(atan(sinθ, cosθ))   # ∈ (-180, 180]
                    end
                end

                # ── δ: signed deviation from expected direction [degrees] ──────
                # Current direction = displacement over window W ending at t
                # Expected direction = mean total displacement of agent's group
                # Sign: positive = anti-clockwise, negative = clockwise
                t_w = t - W
                if t_w >= 1 &&
                   all(isfinite.(pos[t,   i, :])) &&
                   all(isfinite.(pos[t_w, i, :]))
                    exp_dir = i in gA_set ? exp_dir_A : exp_dir_B
                    if all(isfinite.(exp_dir))
                        d_cur  = pos[t, i, :] .- pos[t_w, i, :]
                        n_cur  = norm(d_cur)
                        if n_cur > 1e-9
                            u_cur = d_cur ./ n_cur
                            # signed angle: cross product gives sin, dot gives cos
                            sinδ = exp_dir[1]*u_cur[2] - exp_dir[2]*u_cur[1]
                            cosδ = clamp(dot(exp_dir, u_cur), -1.0, 1.0)
                            DELTA[row_idx, i] = rad2deg(atan(sinδ, cosδ))  # ∈ (-180, 180]
                        end
                    end
                end

                # ── Density ───────────────────────────────────────────────────
                grp_i   = i in gA_set ? gA : gB
                other_i = i in gA_set ? gB : gA

                for method in METHODS
                    fov = parse_fov_tag(method)
                    if fov === nothing
                        nbrs = method == "all"   ? all_agents :
                               method == "intra" ? grp_i      :
                               isempty(other_i)  ? Int[]      : other_i
                        RHO[method][row_idx, i] = isempty(nbrs) ? NaN :
                            density_agent(pos_t, i, nbrs, epsilon)
                    else
                        RHO[method][row_idx, i] = density_agent_fov(
                            pos_t, pos_prev, i, all_agents, epsilon,
                            fov.half_deg, fov.c
                        )
                    end
                end
            end
        end

        # ── Acceleration: (v[t+1] - v[t]) * fps  (NaN for last row) ─────────────
        for row_idx in 1:(n_rows-1)
            for i in all_agents
                if isfinite(V[row_idx+1, i]) && isfinite(V[row_idx, i])
                    ACC[row_idx, i] = (V[row_idx+1, i] - V[row_idx, i]) * fps
                end
            end
        end

        # Save one CSV per method, in csv_data/<method>/Angle_α/
        # Columns: t_prime, v_i, acc_i, rho_i, dangle_i, delta_i  (angles in degrees, signed)
        out_fname = replace(fname, ".csv" => "_vrho.csv")
        for method in METHODS
            dir = joinpath(base_dir, method, "Angle_$α")
            mkpath(dir)
            out = DataFrame(t_prime = t_prime_col)
            for i in 1:n_agents
                out[!, "v_$i"]      = V[:, i]
                out[!, "acc_$i"]    = ACC[:, i]
                out[!, "rho_$i"]    = RHO[method][:, i]
                out[!, "dangle_$i"] = DANGLE[:, i]
                out[!, "delta_$i"]  = DELTA[:, i]
            end
            CSV.write(joinpath(dir, out_fname), out)
        end

        println("Saved $out_fname  (α=$α, agents=$n_agents, rows=$n_rows)")
    end

    println("\nDone. Data in: $base_dir")
end

compute_vrho()