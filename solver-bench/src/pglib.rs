//! PGLib-OPF (Power Grid Library) benchmark infrastructure.
//!
//! Downloads and runs SOCP relaxations of AC-OPF problems from
//! https://github.com/power-grid-lib/pglib-opf
//!
//! The SOCP relaxation transforms the nonconvex AC power flow equations
//! into a convex second-order cone program.

use anyhow::{bail, Context, Result};
use solver_core::linalg::sparse;
use solver_core::{solve, ConeSpec, ProblemData, SolveStatus, SolverSettings};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

const PGLIB_BASE_URL: &str = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master";

/// Available PGLib-OPF test cases (MATPOWER format).
/// These are selected for SOCP relaxation suitability.
pub const PGLIB_PROBLEMS: &[&str] = &[
    // IEEE standard test cases (classic benchmarks)
    "pglib_opf_case14_ieee",  // 14 buses, 5 generators
    "pglib_opf_case30_ieee",  // 30 buses, 6 generators
    "pglib_opf_case57_ieee",  // 57 buses, 7 generators
    "pglib_opf_case118_ieee", // 118 buses, 54 generators
    "pglib_opf_case300_ieee", // 300 buses, 69 generators
    // Small synthetic cases
    "pglib_opf_case3_lmbd", // 3-bus, Lesieutre-Molzahn-DeMarco
    "pglib_opf_case5_pjm",  // 5-bus PJM test case
    // Grid Optimization Competition (GOC) cases
    "pglib_opf_case24_ieee_rts", // 24-bus IEEE RTS
    "pglib_opf_case73_ieee_rts", // 73-bus IEEE RTS (3-area)
    // European grids (PEGASE)
    "pglib_opf_case89_pegase",   // 89-bus PEGASE
    "pglib_opf_case179_goc",     // 179-bus GOC
    "pglib_opf_case240_pserc",   // 240-bus PSERC
    "pglib_opf_case500_goc",     // 500-bus GOC
    "pglib_opf_case1354_pegase", // 1354-bus PEGASE
    "pglib_opf_case2869_pegase", // 2869-bus PEGASE
];

/// MATPOWER bus data (parsed from mpc.bus matrix).
#[derive(Debug, Clone)]
pub struct Bus {
    pub id: usize,       // Bus number
    pub bus_type: usize, // 1=PQ, 2=PV, 3=ref
    pub pd: f64,         // Active power demand (MW)
    pub qd: f64,         // Reactive power demand (MVAr)
    pub gs: f64,         // Shunt conductance (MW at V=1 pu)
    pub bs: f64,         // Shunt susceptance (MVAr at V=1 pu)
    pub vm: f64,         // Voltage magnitude (pu)
    pub va: f64,         // Voltage angle (degrees)
    pub base_kv: f64,    // Base voltage (kV)
    pub vmax: f64,       // Maximum voltage magnitude (pu)
    pub vmin: f64,       // Minimum voltage magnitude (pu)
}

/// MATPOWER generator data (parsed from mpc.gen matrix).
#[derive(Debug, Clone)]
pub struct Generator {
    pub bus: usize,   // Bus number
    pub pg: f64,      // Active power output (MW)
    pub qg: f64,      // Reactive power output (MVAr)
    pub qmax: f64,    // Maximum reactive output (MVAr)
    pub qmin: f64,    // Minimum reactive output (MVAr)
    pub vg: f64,      // Voltage setpoint (pu)
    pub mbase: f64,   // Total MVA base
    pub status: bool, // Machine status (true=in-service)
    pub pmax: f64,    // Maximum active output (MW)
    pub pmin: f64,    // Minimum active output (MW)
}

/// MATPOWER branch data (parsed from mpc.branch matrix).
#[derive(Debug, Clone)]
pub struct Branch {
    pub from_bus: usize, // "From" bus number
    pub to_bus: usize,   // "To" bus number
    pub r: f64,          // Resistance (pu)
    pub x: f64,          // Reactance (pu)
    pub b: f64,          // Total line charging susceptance (pu)
    pub rate_a: f64,     // Long-term rating (MVA)
    pub rate_b: f64,     // Short-term rating (MVA)
    pub rate_c: f64,     // Emergency rating (MVA)
    pub tap: f64,        // Transformer off-nominal turns ratio
    pub shift: f64,      // Transformer phase shift angle (degrees)
    pub status: bool,    // Branch status (true=in-service)
    pub angmin: f64,     // Minimum angle difference (degrees)
    pub angmax: f64,     // Maximum angle difference (degrees)
}

/// MATPOWER generator cost data (parsed from mpc.gencost matrix).
#[derive(Debug, Clone)]
pub struct GenCost {
    pub model: usize,   // Cost model: 1=piecewise linear, 2=polynomial
    pub startup: f64,   // Startup cost ($)
    pub shutdown: f64,  // Shutdown cost ($)
    pub n_cost: usize,  // Number of cost coefficients
    pub cost: Vec<f64>, // Cost coefficients (highest order first for polynomial)
}

/// Parsed MATPOWER case data.
#[derive(Debug, Clone)]
pub struct MatpowerCase {
    pub name: String,
    pub base_mva: f64,
    pub buses: Vec<Bus>,
    pub generators: Vec<Generator>,
    pub branches: Vec<Branch>,
    pub gencost: Vec<GenCost>,
    /// Map from bus ID to index in buses vector
    pub bus_idx: HashMap<usize, usize>,
}

/// Result of running a single PGLib benchmark.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub n_buses: usize,
    pub n_gens: usize,
    pub n_branches: usize,
    pub n: usize,
    pub m: usize,
    pub nnz: usize,
    pub status: SolveStatus,
    pub iterations: usize,
    pub obj_val: f64,
    pub mu: f64,
    pub solve_time_ms: f64,
    pub error: Option<String>,
}

/// Summary statistics for benchmark suite.
#[derive(Debug, Clone)]
pub struct BenchmarkSummary {
    pub total: usize,
    pub optimal: usize,
    pub max_iters: usize,
    pub numerical_error: usize,
    pub other: usize,
    pub parse_errors: usize,
    pub avg_iters: f64,
    pub avg_time_ms: f64,
}

/// Get the cache directory for PGLib problems.
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/minix-bench/pglib")
}

/// Download a MATPOWER file from PGLib.
fn download_matpower(name: &str) -> Result<PathBuf> {
    let cache_dir = get_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    let m_path = cache_dir.join(format!("{}.m", name));

    // Check if already cached
    if m_path.exists() {
        return Ok(m_path);
    }

    // Download from GitHub
    let url = format!("{}/{}.m", PGLIB_BASE_URL, name);

    eprintln!("Downloading {}...", url);

    let output = Command::new("curl")
        .args(["-sL", "--max-time", "60", "-o"])
        .arg(&m_path)
        .arg(&url)
        .output()
        .context("Failed to run curl")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("curl failed: {}", stderr);
    }

    // Check if file was downloaded
    if !m_path.exists() || fs::metadata(&m_path)?.len() == 0 {
        bail!("Download failed: empty or missing file");
    }

    Ok(m_path)
}

/// Parse a MATPOWER .m file.
pub fn parse_matpower(path: &PathBuf) -> Result<MatpowerCase> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut case = MatpowerCase {
        name: path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        base_mva: 100.0,
        buses: Vec::new(),
        generators: Vec::new(),
        branches: Vec::new(),
        gencost: Vec::new(),
        bus_idx: HashMap::new(),
    };

    #[derive(PartialEq)]
    enum Section {
        None,
        Bus,
        Gen,
        Branch,
        GenCost,
    }

    let mut section = Section::None;

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        // Check for section headers
        if line.contains("mpc.baseMVA") {
            if let Some(eq_pos) = line.find('=') {
                let val_str = line[eq_pos + 1..].trim().trim_end_matches(';');
                case.base_mva = val_str.parse().unwrap_or(100.0);
            }
            continue;
        }

        if line.contains("mpc.bus") && line.contains('=') && !line.contains("mpc.bus_name") {
            section = Section::Bus;
            continue;
        }
        if line.contains("mpc.gen") && line.contains('=') && !line.contains("mpc.gencost") {
            section = Section::Gen;
            continue;
        }
        if line.contains("mpc.branch") && line.contains('=') {
            section = Section::Branch;
            continue;
        }
        if line.contains("mpc.gencost") && line.contains('=') {
            section = Section::GenCost;
            continue;
        }

        // Check for section end
        if line.contains("];") {
            section = Section::None;
            continue;
        }

        // Skip function definition and other non-data lines
        if line.starts_with("function") || line.contains("mpc.version") {
            continue;
        }

        // Parse data row
        let row = parse_data_row(line);
        if row.is_empty() {
            continue;
        }

        match section {
            Section::Bus if row.len() >= 13 => {
                let bus = Bus {
                    id: row[0] as usize,
                    bus_type: row[1] as usize,
                    pd: row[2],
                    qd: row[3],
                    gs: row[4],
                    bs: row[5],
                    vm: row[7],
                    va: row[8],
                    base_kv: row[9],
                    vmax: row[11],
                    vmin: row[12],
                };
                case.bus_idx.insert(bus.id, case.buses.len());
                case.buses.push(bus);
            }
            Section::Gen if row.len() >= 10 => {
                let gen = Generator {
                    bus: row[0] as usize,
                    pg: row[1],
                    qg: row[2],
                    qmax: row[3],
                    qmin: row[4],
                    vg: row[5],
                    mbase: row[6],
                    status: row[7] > 0.5,
                    pmax: row[8],
                    pmin: row[9],
                };
                case.generators.push(gen);
            }
            Section::Branch if row.len() >= 13 => {
                let branch = Branch {
                    from_bus: row[0] as usize,
                    to_bus: row[1] as usize,
                    r: row[2],
                    x: row[3],
                    b: row[4],
                    rate_a: row[5],
                    rate_b: row[6],
                    rate_c: row[7],
                    tap: if row[8] == 0.0 { 1.0 } else { row[8] },
                    shift: row[9],
                    status: row[10] > 0.5,
                    angmin: row[11],
                    angmax: row[12],
                };
                case.branches.push(branch);
            }
            Section::GenCost if row.len() >= 4 => {
                let model = row[0] as usize;
                let n_cost = row[3] as usize;
                let cost: Vec<f64> = row.iter().skip(4).take(n_cost).copied().collect();
                let gencost = GenCost {
                    model,
                    startup: row[1],
                    shutdown: row[2],
                    n_cost,
                    cost,
                };
                case.gencost.push(gencost);
            }
            _ => {}
        }
    }

    if case.buses.is_empty() {
        bail!("No bus data found in MATPOWER file");
    }

    Ok(case)
}

/// Parse a data row from MATPOWER format.
fn parse_data_row(line: &str) -> Vec<f64> {
    let mut values = Vec::new();

    // Remove inline comments (% to end of line)
    let line_no_comment = if let Some(pos) = line.find('%') {
        &line[..pos]
    } else {
        line
    };

    // Remove trailing semicolon and brackets (trim first to handle trailing spaces)
    let clean = line_no_comment
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(';')
        .trim_end_matches(']')
        .trim();

    // Split by whitespace or tabs
    for part in clean.split_whitespace() {
        // Skip non-numeric tokens
        if part.is_empty() {
            continue;
        }

        // Try to parse as number
        if let Ok(val) = part.parse::<f64>() {
            values.push(val);
        }
    }

    values
}

/// Build SOCP relaxation of AC-OPF from MATPOWER case.
///
/// Variables (per unit):
/// - v_i = squared voltage magnitude at bus i (n_bus)
/// - p_g,i = active generation at gen i (n_gen)
/// - q_g,i = reactive generation at gen i (n_gen)
/// - p_ij = active power flow from i to j on branch (n_branch)
/// - q_ij = reactive power flow from i to j on branch (n_branch)
/// - l_ij = squared current magnitude on branch (n_branch)
///
/// SOCP constraint for each branch:
///   ||(2*p_ij, 2*q_ij, l_ij - v_i)||_2 <= l_ij + v_i
pub fn build_socp_relaxation(case: &MatpowerCase) -> Result<ProblemData> {
    let n_bus = case.buses.len();
    let n_gen = case.generators.iter().filter(|g| g.status).count();
    let n_branch = case.branches.iter().filter(|b| b.status).count();

    // Variable indices
    let v_off = 0; // v_i: n_bus
    let pg_off = n_bus; // p_g: n_gen
    let qg_off = pg_off + n_gen; // q_g: n_gen
    let pf_off = qg_off + n_gen; // p_ij (from): n_branch
    let qf_off = pf_off + n_branch; // q_ij (from): n_branch
    let pt_off = qf_off + n_branch; // p_ij (to): n_branch
    let qt_off = pt_off + n_branch; // q_ij (to): n_branch
    let l_off = qt_off + n_branch; // l_ij: n_branch

    let n_vars = l_off + n_branch;

    // Build generator bus mapping
    let mut gen_at_bus: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut gen_idx = 0;
    for gen in &case.generators {
        if gen.status {
            gen_at_bus.entry(gen.bus).or_default().push(gen_idx);
            gen_idx += 1;
        }
    }

    // Active branches with indices
    let active_branches: Vec<(usize, &Branch)> = case
        .branches
        .iter()
        .filter(|b| b.status)
        .enumerate()
        .collect();

    // Constraints:
    // 1. Power balance at each bus (2 * n_bus equality constraints)
    // 2. Voltage magnitude bounds (2 * n_bus inequality constraints)
    // 3. Generator output bounds (4 * n_gen inequality constraints)
    // 4. Branch power flow limits (n_branch inequality constraints, if rate_a > 0)
    // 5. SOCP relaxation (4 * n_branch for each SOC)

    let n_eq = 2 * n_bus; // P and Q balance
    let n_vbnd = 2 * n_bus; // vmin, vmax
    let n_gbnd = 4 * n_gen; // pmin, pmax, qmin, qmax
    let n_thermal = active_branches
        .iter()
        .filter(|(_, b)| b.rate_a > 0.0)
        .count();
    let n_soc = 4 * n_branch;

    let total_m = n_eq + n_vbnd + n_gbnd + n_thermal + n_soc;

    let mut triplets = Vec::new();
    let mut b_vec = vec![0.0; total_m];

    // =========================================================================
    // Objective: Minimize generation cost (typically c2*pg^2 + c1*pg + c0)
    // For linear cost, just use c1*pg
    // =========================================================================
    let mut q = vec![0.0; n_vars];
    let mut gen_idx = 0;
    for (i, gen) in case.generators.iter().enumerate() {
        if !gen.status {
            continue;
        }

        // Get cost coefficients (if available)
        if i < case.gencost.len() {
            let gc = &case.gencost[i];
            if gc.model == 2 && !gc.cost.is_empty() {
                // Polynomial cost: last coefficient is linear term
                // Cost is c[0]*p^n + c[1]*p^(n-1) + ... + c[n]
                // For quadratic (n=3): c[0]*p^2 + c[1]*p + c[2]
                // Linear term is second-to-last for n=2, last for n=1
                let linear_coef = if gc.cost.len() >= 2 {
                    gc.cost[gc.cost.len() - 2] // Linear coefficient
                } else if gc.cost.len() == 1 {
                    gc.cost[0]
                } else {
                    0.0
                };
                q[pg_off + gen_idx] = linear_coef / case.base_mva;
            }
        }
        gen_idx += 1;
    }

    // =========================================================================
    // Power balance constraints (equality)
    // P: sum(pg at bus i) - pd_i - sum(pf leaving i) + sum(pt entering i) = 0
    // Q: sum(qg at bus i) - qd_i - sum(qf leaving i) + sum(qt entering i) = 0
    // =========================================================================
    for (bus_i, bus) in case.buses.iter().enumerate() {
        let p_row = bus_i;
        let q_row = n_bus + bus_i;

        // Generator injections at this bus
        if let Some(gens) = gen_at_bus.get(&bus.id) {
            for &g_idx in gens {
                triplets.push((p_row, pg_off + g_idx, 1.0));
                triplets.push((q_row, qg_off + g_idx, 1.0));
            }
        }

        // Shunt elements (constant power consumed based on voltage)
        // P_shunt = Gs * v, Q_shunt = -Bs * v
        if bus.gs.abs() > 1e-10 {
            triplets.push((p_row, v_off + bus_i, -bus.gs / case.base_mva));
        }
        if bus.bs.abs() > 1e-10 {
            triplets.push((q_row, v_off + bus_i, bus.bs / case.base_mva));
        }

        // Branch flows
        for (br_idx, branch) in &active_branches {
            let from_i = *case.bus_idx.get(&branch.from_bus).unwrap();
            let to_i = *case.bus_idx.get(&branch.to_bus).unwrap();

            if from_i == bus_i {
                // Power leaving this bus
                triplets.push((p_row, pf_off + br_idx, -1.0));
                triplets.push((q_row, qf_off + br_idx, -1.0));
            }
            if to_i == bus_i {
                // Power entering this bus
                triplets.push((p_row, pt_off + br_idx, 1.0));
                triplets.push((q_row, qt_off + br_idx, 1.0));
            }
        }

        // RHS: load demand (converted to per-unit)
        b_vec[p_row] = bus.pd / case.base_mva;
        b_vec[q_row] = bus.qd / case.base_mva;
    }

    // =========================================================================
    // Voltage bounds: vmin^2 <= v_i <= vmax^2
    // -v_i + s = -vmin^2 (s >= 0 means v >= vmin^2)
    // v_i + s = vmax^2 (s >= 0 means v <= vmax^2)
    // =========================================================================
    let vbnd_off = n_eq;
    for (bus_i, bus) in case.buses.iter().enumerate() {
        let vmin2 = bus.vmin * bus.vmin;
        let vmax2 = bus.vmax * bus.vmax;

        // v >= vmin^2
        triplets.push((vbnd_off + 2 * bus_i, v_off + bus_i, -1.0));
        b_vec[vbnd_off + 2 * bus_i] = -vmin2;

        // v <= vmax^2
        triplets.push((vbnd_off + 2 * bus_i + 1, v_off + bus_i, 1.0));
        b_vec[vbnd_off + 2 * bus_i + 1] = vmax2;
    }

    // =========================================================================
    // Generator bounds: pmin <= pg <= pmax, qmin <= qg <= qmax
    // =========================================================================
    let gbnd_off = vbnd_off + n_vbnd;
    let mut gen_idx = 0;
    for gen in &case.generators {
        if !gen.status {
            continue;
        }

        let pmin = gen.pmin / case.base_mva;
        let pmax = gen.pmax / case.base_mva;
        let qmin = gen.qmin / case.base_mva;
        let qmax = gen.qmax / case.base_mva;

        // pg >= pmin
        triplets.push((gbnd_off + 4 * gen_idx, pg_off + gen_idx, -1.0));
        b_vec[gbnd_off + 4 * gen_idx] = -pmin;

        // pg <= pmax
        triplets.push((gbnd_off + 4 * gen_idx + 1, pg_off + gen_idx, 1.0));
        b_vec[gbnd_off + 4 * gen_idx + 1] = pmax;

        // qg >= qmin
        triplets.push((gbnd_off + 4 * gen_idx + 2, qg_off + gen_idx, -1.0));
        b_vec[gbnd_off + 4 * gen_idx + 2] = -qmin;

        // qg <= qmax
        triplets.push((gbnd_off + 4 * gen_idx + 3, qg_off + gen_idx, 1.0));
        b_vec[gbnd_off + 4 * gen_idx + 3] = qmax;

        gen_idx += 1;
    }

    // =========================================================================
    // Thermal limits: pf^2 + qf^2 <= rate_a^2
    // This is an SOC constraint, but for simplicity we use a relaxed linear bound
    // |pf| + |qf| <= rate_a (conservative approximation)
    // Or we could add as separate SOC cones
    // For now, skip this and rely on the main SOCP constraint
    // =========================================================================
    let thermal_off = gbnd_off + n_gbnd;
    let mut thermal_idx = 0;
    for (br_idx, branch) in &active_branches {
        if branch.rate_a > 0.0 {
            let rate = branch.rate_a / case.base_mva;
            // |pf| <= rate (simplified - proper version needs SOC)
            // pf + s = rate, s >= 0 => pf <= rate
            triplets.push((thermal_off + thermal_idx, pf_off + br_idx, 1.0));
            b_vec[thermal_off + thermal_idx] = rate;
            thermal_idx += 1;
        }
    }

    // =========================================================================
    // SOCP relaxation for each branch
    // ||(2*pf, 2*qf, l - v_from)||_2 <= l + v_from
    // Reformulated: (l + v, 2*pf, 2*qf, l - v) in SOC
    // s_0 = l + v => -l - v + s_0 = 0
    // s_1 = 2*pf => -2*pf + s_1 = 0
    // s_2 = 2*qf => -2*qf + s_2 = 0
    // s_3 = l - v => -l + v + s_3 = 0
    // =========================================================================
    let soc_off = thermal_off + n_thermal;
    for (br_idx, branch) in &active_branches {
        let from_i = *case.bus_idx.get(&branch.from_bus).unwrap();
        let row_base = soc_off + 4 * br_idx;

        // s_0 = l + v_from
        triplets.push((row_base, l_off + br_idx, -1.0));
        triplets.push((row_base, v_off + from_i, -1.0));

        // s_1 = 2*pf
        triplets.push((row_base + 1, pf_off + br_idx, -2.0));

        // s_2 = 2*qf
        triplets.push((row_base + 2, qf_off + br_idx, -2.0));

        // s_3 = l - v_from
        triplets.push((row_base + 3, l_off + br_idx, -1.0));
        triplets.push((row_base + 3, v_off + from_i, 1.0));
    }

    let a = sparse::from_triplets(total_m, n_vars, triplets);

    // Build cone specification
    let mut cones = vec![
        ConeSpec::Zero { dim: n_eq },
        ConeSpec::NonNeg {
            dim: n_vbnd + n_gbnd + n_thermal,
        },
    ];

    for _ in 0..n_branch {
        cones.push(ConeSpec::Soc { dim: 4 });
    }

    Ok(ProblemData {
        P: None,
        q,
        A: a,
        b: b_vec,
        cones,
        var_bounds: None,
        integrality: None,
    })
}

/// Load a PGLib problem.
pub fn load_problem(name: &str) -> Result<MatpowerCase> {
    let path = download_matpower(name)?;
    parse_matpower(&path)
}

/// Run a single PGLib benchmark.
pub fn run_single(name: &str, settings: &SolverSettings) -> BenchmarkResult {
    // Load and parse MATPOWER case
    let case = match load_problem(name) {
        Ok(c) => c,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n_buses: 0,
                n_gens: 0,
                n_branches: 0,
                n: 0,
                m: 0,
                nnz: 0,
                status: SolveStatus::NumericalError,
                iterations: 0,
                obj_val: 0.0,
                mu: 0.0,
                solve_time_ms: 0.0,
                error: Some(format!("Parse error: {}", e)),
            };
        }
    };

    let n_buses = case.buses.len();
    let n_gens = case.generators.iter().filter(|g| g.status).count();
    let n_branches = case.branches.iter().filter(|b| b.status).count();

    // Build SOCP relaxation
    let prob = match build_socp_relaxation(&case) {
        Ok(p) => p,
        Err(e) => {
            return BenchmarkResult {
                name: name.to_string(),
                n_buses,
                n_gens,
                n_branches,
                n: 0,
                m: 0,
                nnz: 0,
                status: SolveStatus::NumericalError,
                iterations: 0,
                obj_val: 0.0,
                mu: 0.0,
                solve_time_ms: 0.0,
                error: Some(format!("SOCP formulation error: {}", e)),
            };
        }
    };

    let n = prob.num_vars();
    let m = prob.num_constraints();
    let nnz = prob.A.nnz();

    // Solve
    let start = Instant::now();
    let result = solve(&prob, settings);
    let elapsed = start.elapsed();

    match result {
        Ok(res) => BenchmarkResult {
            name: name.to_string(),
            n_buses,
            n_gens,
            n_branches,
            n,
            m,
            nnz,
            status: res.status,
            iterations: res.info.iters,
            obj_val: res.obj_val,
            mu: res.info.mu,
            solve_time_ms: elapsed.as_secs_f64() * 1000.0,
            error: None,
        },
        Err(e) => BenchmarkResult {
            name: name.to_string(),
            n_buses,
            n_gens,
            n_branches,
            n,
            m,
            nnz,
            status: SolveStatus::NumericalError,
            iterations: 0,
            obj_val: 0.0,
            mu: 0.0,
            solve_time_ms: elapsed.as_secs_f64() * 1000.0,
            error: Some(format!("Solve error: {}", e)),
        },
    }
}

/// Run the full PGLib suite.
pub fn run_full_suite(settings: &SolverSettings, limit: Option<usize>) -> Vec<BenchmarkResult> {
    let problems: Vec<_> = if let Some(limit) = limit {
        PGLIB_PROBLEMS.iter().take(limit).collect()
    } else {
        PGLIB_PROBLEMS.iter().collect()
    };

    let mut results = Vec::with_capacity(problems.len());

    for (i, name) in problems.iter().enumerate() {
        eprint!("[{}/{}] {}... ", i + 1, problems.len(), name);

        let result = run_single(name, settings);

        if let Some(ref err) = result.error {
            eprintln!("ERROR: {}", err);
        } else {
            eprintln!(
                "{:?} in {} iters, {:.1}ms ({}B/{}G/{}L)",
                result.status,
                result.iterations,
                result.solve_time_ms,
                result.n_buses,
                result.n_gens,
                result.n_branches
            );
        }

        results.push(result);
    }

    results
}

/// Compute summary statistics.
pub fn compute_summary(results: &[BenchmarkResult]) -> BenchmarkSummary {
    let mut summary = BenchmarkSummary {
        total: results.len(),
        optimal: 0,
        max_iters: 0,
        numerical_error: 0,
        other: 0,
        parse_errors: 0,
        avg_iters: 0.0,
        avg_time_ms: 0.0,
    };

    let mut total_iters = 0;
    let mut total_time = 0.0;
    let mut solved_count = 0;

    for r in results {
        if r.error.is_some() {
            summary.parse_errors += 1;
            continue;
        }

        match r.status {
            SolveStatus::Optimal => summary.optimal += 1,
            SolveStatus::MaxIters => summary.max_iters += 1,
            SolveStatus::NumericalError => summary.numerical_error += 1,
            _ => summary.other += 1,
        }

        total_iters += r.iterations;
        total_time += r.solve_time_ms;
        solved_count += 1;
    }

    if solved_count > 0 {
        summary.avg_iters = total_iters as f64 / solved_count as f64;
        summary.avg_time_ms = total_time / solved_count as f64;
    }

    summary
}

/// Print results table.
pub fn print_results_table(results: &[BenchmarkResult]) {
    println!("\n{:-<110}", "");
    println!(
        "{:<30} {:>6} {:>5} {:>6} {:>8} {:>10} {:>8} {:>12} {:>10}",
        "Problem", "Buses", "Gens", "Lines", "n", "Status", "Iters", "Objective", "Time(ms)"
    );
    println!("{:-<110}", "");

    for r in results {
        let status_str = if r.error.is_some() {
            "ERROR".to_string()
        } else {
            format!("{:?}", r.status)
        };

        println!(
            "{:<30} {:>6} {:>5} {:>6} {:>8} {:>10} {:>8} {:>12.4e} {:>10.1}",
            r.name,
            r.n_buses,
            r.n_gens,
            r.n_branches,
            r.n,
            status_str,
            r.iterations,
            r.obj_val,
            r.solve_time_ms
        );
    }

    println!("{:-<110}", "");
}

/// Print summary.
pub fn print_summary(summary: &BenchmarkSummary) {
    println!("\nPGLib-OPF Benchmark Summary");
    println!("===========================");
    println!("Total problems:     {}", summary.total);
    println!("Parse errors:       {}", summary.parse_errors);
    println!();
    println!(
        "Optimal:            {} ({:.1}%)",
        summary.optimal,
        100.0 * summary.optimal as f64 / (summary.total - summary.parse_errors).max(1) as f64
    );
    println!("Max iterations:     {}", summary.max_iters);
    println!("Numerical error:    {}", summary.numerical_error);
    println!("Other:              {}", summary.other);
    println!();
    println!("Avg iterations:     {:.1}", summary.avg_iters);
    println!("Avg solve time:     {:.1} ms", summary.avg_time_ms);
}
