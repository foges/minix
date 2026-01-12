//! Unified verbosity and diagnostics configuration.
//!
//! Provides a single `MINIX_VERBOSE` environment variable with levels 0-4:
//! - 0: Silent (no output)
//! - 1: Normal (solve summary only) [default]
//! - 2: Verbose (iteration table, recovery messages)
//! - 3: Debug (detailed per-iteration logging, step info)
//! - 4: Trace (all diagnostics including cone-specific debug)

use std::env;

/// Verbosity level for solver output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum VerbosityLevel {
    /// No output at all
    Silent = 0,
    /// Solve summary only (status, objective, time)
    Normal = 1,
    /// Iteration table with residuals, recovery messages
    Verbose = 2,
    /// Detailed per-iteration logging, step sizes, mu values
    Debug = 3,
    /// All diagnostics including cone-specific debug output
    Trace = 4,
}

impl VerbosityLevel {
    /// Parse from integer (0-4), clamping to valid range.
    pub fn from_int(level: u8) -> Self {
        match level {
            0 => VerbosityLevel::Silent,
            1 => VerbosityLevel::Normal,
            2 => VerbosityLevel::Verbose,
            3 => VerbosityLevel::Debug,
            _ => VerbosityLevel::Trace,
        }
    }

    /// Check if this level enables the given level.
    #[inline]
    pub fn enables(&self, level: VerbosityLevel) -> bool {
        *self >= level
    }

    /// Convenience: is verbose or higher?
    #[inline]
    pub fn is_verbose(&self) -> bool {
        *self >= VerbosityLevel::Verbose
    }

    /// Convenience: is debug or higher?
    #[inline]
    pub fn is_debug(&self) -> bool {
        *self >= VerbosityLevel::Debug
    }

    /// Convenience: is trace?
    #[inline]
    pub fn is_trace(&self) -> bool {
        *self >= VerbosityLevel::Trace
    }
}

impl Default for VerbosityLevel {
    fn default() -> Self {
        VerbosityLevel::Normal
    }
}

/// Diagnostics configuration for the solver.
///
/// Configuration is determined by `MINIX_VERBOSE` environment variable:
/// - `MINIX_VERBOSE=0` - Silent
/// - `MINIX_VERBOSE=1` - Normal (default)
/// - `MINIX_VERBOSE=2` - Verbose (iteration table)
/// - `MINIX_VERBOSE=3` - Debug (detailed logging)
/// - `MINIX_VERBOSE=4` - Trace (all diagnostics)
///
/// For backward compatibility, these legacy env vars are also checked:
/// - `MINIX_DIAGNOSTICS=1` sets level to Verbose (2)
/// - `MINIX_ITER_LOG=1` sets level to Debug (3)
/// - `MINIX_QFORPLAN_DIAG=1` sets level to Trace (4)
#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    /// Current verbosity level
    pub level: VerbosityLevel,
    /// Log every N iterations (only applies at Debug level and above)
    pub every: usize,
    /// Include KKT residuals in debug output
    pub print_kkt_residuals: bool,
}

impl DiagnosticsConfig {
    /// Create from environment variables.
    ///
    /// Priority order:
    /// 1. `MINIX_VERBOSE=N` (0-4)
    /// 2. Legacy vars: `MINIX_QFORPLAN_DIAG` → 4, `MINIX_ITER_LOG` → 3, `MINIX_DIAGNOSTICS` → 2
    /// 3. Default: Normal (1)
    pub fn from_env() -> Self {
        // Check new unified env var first
        let level = if let Ok(v) = env::var("MINIX_VERBOSE") {
            match v.parse::<u8>() {
                Ok(n) => VerbosityLevel::from_int(n),
                Err(_) => {
                    // Handle string values
                    match v.to_lowercase().as_str() {
                        "silent" | "off" | "false" => VerbosityLevel::Silent,
                        "normal" => VerbosityLevel::Normal,
                        "verbose" | "v" => VerbosityLevel::Verbose,
                        "debug" | "vv" => VerbosityLevel::Debug,
                        "trace" | "vvv" | "all" => VerbosityLevel::Trace,
                        _ => VerbosityLevel::Normal,
                    }
                }
            }
        } else {
            // Legacy env var fallback (check most specific first)
            if env::var("MINIX_QFORPLAN_DIAG").is_ok() {
                VerbosityLevel::Trace
            } else if env::var("MINIX_ITER_LOG").is_ok() {
                VerbosityLevel::Debug
            } else if env::var("MINIX_DIAGNOSTICS").is_ok() {
                // Check if explicitly disabled
                match env::var("MINIX_DIAGNOSTICS") {
                    Ok(v) if v == "0" || v.to_lowercase() == "false" => VerbosityLevel::Normal,
                    _ => VerbosityLevel::Verbose,
                }
            } else {
                VerbosityLevel::Normal
            }
        };

        let every = env::var("MINIX_VERBOSE_EVERY")
            .or_else(|_| env::var("MINIX_DIAGNOSTICS_EVERY"))
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1);

        let print_kkt_residuals = env::var("MINIX_VERBOSE_KKT")
            .or_else(|_| env::var("MINIX_DIAGNOSTICS_KKT"))
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        Self { level, every, print_kkt_residuals }
    }

    /// Create with explicit verbosity level.
    pub fn with_level(level: VerbosityLevel) -> Self {
        Self {
            level,
            every: 1,
            print_kkt_residuals: true,
        }
    }

    /// Create silent config (no output).
    pub fn silent() -> Self {
        Self::with_level(VerbosityLevel::Silent)
    }

    /// Create from SolverSettings verbose flag.
    /// If settings.verbose is true, use Verbose level.
    /// Otherwise, check environment.
    pub fn from_settings_verbose(verbose: bool) -> Self {
        let mut config = Self::from_env();
        if verbose && config.level < VerbosityLevel::Verbose {
            config.level = VerbosityLevel::Verbose;
        }
        config
    }

    // --- Convenience methods ---

    /// Is any output enabled?
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.level > VerbosityLevel::Silent
    }

    /// Should log at Verbose level (iteration table)?
    #[inline]
    pub fn is_verbose(&self) -> bool {
        self.level >= VerbosityLevel::Verbose
    }

    /// Should log at Debug level (detailed per-iteration)?
    #[inline]
    pub fn is_debug(&self) -> bool {
        self.level >= VerbosityLevel::Debug
    }

    /// Should log at Trace level (all diagnostics)?
    #[inline]
    pub fn is_trace(&self) -> bool {
        self.level >= VerbosityLevel::Trace
    }

    /// Should log this iteration at Debug level?
    #[inline]
    pub fn should_log_iter(&self, iter: usize) -> bool {
        self.is_debug() && (iter % self.every == 0)
    }

    // --- Legacy compatibility ---

    /// Legacy: equivalent to is_verbose()
    #[inline]
    pub fn enabled(&self) -> bool {
        self.is_verbose()
    }

    /// Legacy: equivalent to should_log_iter()
    #[inline]
    pub fn should_log(&self, iter: usize) -> bool {
        self.should_log_iter(iter)
    }
}

impl Default for DiagnosticsConfig {
    fn default() -> Self {
        Self::from_env()
    }
}
