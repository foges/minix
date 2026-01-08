//! Shared test problem definitions for regression and benchmarking.

use solver_core::{ConeSpec, ProblemData};

/// Test problem definition.
#[derive(Clone)]
pub struct TestProblem {
    pub name: &'static str,
    pub problem_class: &'static str,
    pub builder: fn() -> ProblemData,
    pub expected_iterations: Option<usize>,
    pub expected_wallclock_ms: Option<f64>,
    pub expected_to_fail: bool,
    pub source: &'static str,
}

// ============================================================================
// Synthetic LP/SOC Problems
// ============================================================================

fn build_syn_lp_nonneg() -> ProblemData {
    let a = solver_core::linalg::sparse::from_triplets(1, 1, vec![(0, 0, -1.0)]);
    ProblemData {
        P: None,
        q: vec![1.0],
        A: a,
        b: vec![0.0],
        cones: vec![ConeSpec::NonNeg { dim: 1 }],
        var_bounds: None,
        integrality: None,
    }
}

fn build_syn_soc_feas() -> ProblemData {
    let a = solver_core::linalg::sparse::from_triplets(
        2,
        2,
        vec![(0, 0, -1.0), (1, 1, -1.0)],
    );
    ProblemData {
        P: None,
        q: vec![0.0, 0.0],
        A: a,
        b: vec![0.0, 0.0],
        cones: vec![ConeSpec::Soc { dim: 2 }],
        var_bounds: None,
        integrality: None,
    }
}

// TODO: Add real-world SDP and exponential cone problems

// ============================================================================
// Problem Registry
// ============================================================================

pub fn synthetic_test_problems() -> Vec<TestProblem> {
    vec![
        TestProblem {
            name: "SYN_LP_NONNEG",
            problem_class: "LP",
            builder: build_syn_lp_nonneg,
            expected_iterations: Some(5),
            expected_wallclock_ms: None,
            expected_to_fail: false,
            source: "synthetic",
        },
        TestProblem {
            name: "SYN_SOC_FEAS",
            problem_class: "SOCP",
            builder: build_syn_soc_feas,
            expected_iterations: Some(9),
            expected_wallclock_ms: None,
            expected_to_fail: false,
            source: "synthetic",
        },
        // TODO: Add real-world SDP and exponential cone problems
    ]
}

pub fn maros_meszaros_problem_names() -> &'static [&'static str] {
    &[
        // HS problems (tiny)
        "HS21", "HS35", "HS35MOD", "HS51", "HS52", "HS53", "HS76", "HS118", "HS268",
        // Other tiny (<1ms)
        "TAME", "S268", "ZECEVIC2", "LOTSCHD", "QAFIRO",
        // CVXQP family (all 9)
        "CVXQP1_S", "CVXQP2_S", "CVXQP3_S",
        "CVXQP1_M", "CVXQP2_M", "CVXQP3_M",
        "CVXQP1_L", "CVXQP2_L", "CVXQP3_L",
        // DUAL/PRIMAL families (all 16)
        "DUAL1", "DUAL2", "DUAL3", "DUAL4",
        "DUALC1", "DUALC2", "DUALC5", "DUALC8",
        "PRIMAL1", "PRIMAL2", "PRIMAL3", "PRIMAL4",
        "PRIMALC1", "PRIMALC2", "PRIMALC5", "PRIMALC8",
        // AUG family (all 8)
        "AUG2D", "AUG2DC", "AUG2DCQP", "AUG2DQP",
        "AUG3D", "AUG3DC", "AUG3DCQP", "AUG3DQP",
        // CONT family (all 6)
        "CONT-050", "CONT-100", "CONT-101", "CONT-200", "CONT-201", "CONT-300",
        // LISWET family (all 12)
        "LISWET1", "LISWET2", "LISWET3", "LISWET4", "LISWET5", "LISWET6",
        "LISWET7", "LISWET8", "LISWET9", "LISWET10", "LISWET11", "LISWET12",
        // STADAT family (all 3)
        "STADAT1", "STADAT2", "STADAT3",
        // QGROW family (all 3)
        "QGROW7", "QGROW15", "QGROW22",
        // Q* problems that pass with good quality
        "QETAMACR", "QISRAEL",
        "QPCBLEND", "QPCBOEI2", "QPCSTAIR",
        "QRECIPE", "QSC205",
        "QSCSD1", "QSCSD6", "QSCSD8", "QSCTAP1", "QSCTAP2", "QSCTAP3",
        "QSEBA", "QSHARE2B", "QSHELL", "QSIERRA", "QSTAIR", "QSTANDAT",
        // Other medium/large
        "DPKLO1", "DTOC3", "EXDATA", "GOULDQP2", "GOULDQP3",
        "HUES-MOD", "HUESTIS", "KSIP", "LASER",
        "MOSARQP1", "MOSARQP2", "POWELL20",
        "STCQP2", "UBH1", "VALUES", "YAO",
        // BOYD portfolio QPs (~93k vars)
        "BOYD1", "BOYD2",
    ]
}

pub fn maros_meszaros_expected_failures() -> &'static [&'static str] {
    &[
        "Q25FV47", "QADLITTL", "QBANDM", "QBEACONF", "QBORE3D",
        "QBRANDY", "QCAPRI", "QE226", "QFFFFF80", "QFORPLAN",
        "QGFRDXPN", "QPCBOEI1", "QPILOTNO",
        "QSCAGR25", "QSCAGR7", "QSCFXM1", "QSCFXM2", "QSCFXM3",
        "QSCORPIO", "QSCRS8", "QSHARE1B",
        "QSHIP04L", "QSHIP04S", "QSHIP08L", "QSHIP08S",
        "QSHIP12L", "QSHIP12S", "STCQP1",
    ]
}
