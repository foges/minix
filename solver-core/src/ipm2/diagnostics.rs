use std::env;

#[derive(Debug, Clone)]
pub struct DiagnosticsConfig {
    pub enabled: bool,
    pub every: usize,
    pub print_kkt_residuals: bool,
}

impl DiagnosticsConfig {
    pub fn from_env() -> Self {
        let enabled = match env::var("MINIX_DIAGNOSTICS") {
            Ok(v) => v != "0" && v.to_lowercase() != "false",
            Err(_) => false,
        };

        let every = env::var("MINIX_DIAGNOSTICS_EVERY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1);

        let print_kkt_residuals = env::var("MINIX_DIAGNOSTICS_KKT")
            .ok()
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);

        Self { enabled, every, print_kkt_residuals }
    }

    #[inline]
    pub fn should_log(&self, iter: usize) -> bool {
        self.enabled && (iter % self.every == 0)
    }
}

