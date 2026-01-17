use std::time::{Duration, Instant};

#[derive(Debug, Copy, Clone)]
pub enum PerfSection {
    Residuals,
    Scaling,
    KktUpdate,
    Factorization,
    Solve,
    LineSearch,
    StateUpdate,
    Corrector,
    PreStep,
    PostStep,
    Termination,
    LoopEnd,
    Other,
}

#[derive(Debug, Default, Clone)]
pub struct PerfTimers {
    pub residuals: Duration,
    pub scaling: Duration,
    pub kkt_update: Duration,
    pub factorization: Duration,
    pub solve: Duration,
    pub line_search: Duration,
    pub state_update: Duration,
    pub corrector: Duration,
    pub pre_step: Duration,
    pub post_step: Duration,
    pub termination: Duration,
    pub loop_end: Duration,
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
            PerfSection::LineSearch => self.line_search += dt,
            PerfSection::StateUpdate => self.state_update += dt,
            PerfSection::Corrector => self.corrector += dt,
            PerfSection::PreStep => self.pre_step += dt,
            PerfSection::PostStep => self.post_step += dt,
            PerfSection::Termination => self.termination += dt,
            PerfSection::LoopEnd => self.loop_end += dt,
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

