use std::time::{Duration, Instant};

#[derive(Debug, Copy, Clone)]
pub enum PerfSection {
    Residuals,
    Scaling,
    KktUpdate,
    Factorization,
    Solve,
    Termination,
    Other,
}

#[derive(Debug, Default, Clone)]
pub struct PerfTimers {
    pub residuals: Duration,
    pub scaling: Duration,
    pub kkt_update: Duration,
    pub factorization: Duration,
    pub solve: Duration,
    pub termination: Duration,
    pub other: Duration,
}

impl PerfTimers {
    pub fn scoped<'a>(&'a mut self, section: PerfSection) -> PerfGuard<'a> {
        PerfGuard { section, start: Instant::now(), timers: self }
    }

    pub fn add(&mut self, section: PerfSection, dt: Duration) {
        match section {
            PerfSection::Residuals => self.residuals += dt,
            PerfSection::Scaling => self.scaling += dt,
            PerfSection::KktUpdate => self.kkt_update += dt,
            PerfSection::Factorization => self.factorization += dt,
            PerfSection::Solve => self.solve += dt,
            PerfSection::Termination => self.termination += dt,
            PerfSection::Other => self.other += dt,
        }
    }
}

pub struct PerfGuard<'a> {
    section: PerfSection,
    start: Instant,
    timers: &'a mut PerfTimers,
}

impl Drop for PerfGuard<'_> {
    fn drop(&mut self) {
        self.timers.add(self.section, self.start.elapsed());
    }
}

