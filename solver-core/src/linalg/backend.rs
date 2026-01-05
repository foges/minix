use super::qdldl::{QdldlError, QdldlFactorization, QdldlSolver};
use super::sparse::SparseCsc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Qdldl(#[from] QdldlError),
}

pub trait KktBackend {
    type Factorization;

    fn new(n: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self
    where
        Self: Sized;
    fn set_static_reg(&mut self, static_reg: f64) -> Result<(), BackendError>;
    fn static_reg(&self) -> f64;
    fn symbolic_factorization(&mut self, kkt: &SparseCsc) -> Result<(), BackendError>;
    fn numeric_factorization(&mut self, kkt: &SparseCsc) -> Result<Self::Factorization, BackendError>;
    fn solve(&self, factor: &Self::Factorization, rhs: &[f64], sol: &mut [f64]);
    fn dynamic_bumps(&self) -> u64;
}

pub struct QdldlBackend {
    solver: QdldlSolver,
}

impl KktBackend for QdldlBackend {
    type Factorization = QdldlFactorization;

    fn new(n: usize, static_reg: f64, dynamic_reg_min_pivot: f64) -> Self {
        Self {
            solver: QdldlSolver::new(n, static_reg, dynamic_reg_min_pivot),
        }
    }

    fn set_static_reg(&mut self, static_reg: f64) -> Result<(), BackendError> {
        self.solver.set_static_reg(static_reg)?;
        Ok(())
    }

    fn static_reg(&self) -> f64 {
        self.solver.static_reg()
    }

    fn symbolic_factorization(&mut self, kkt: &SparseCsc) -> Result<(), BackendError> {
        self.solver.symbolic_factorization(kkt)?;
        Ok(())
    }

    fn numeric_factorization(&mut self, kkt: &SparseCsc) -> Result<Self::Factorization, BackendError> {
        Ok(self.solver.numeric_factorization(kkt)?)
    }

    fn solve(&self, factor: &Self::Factorization, rhs: &[f64], sol: &mut [f64]) {
        self.solver.solve(factor, rhs, sol);
    }

    fn dynamic_bumps(&self) -> u64 {
        self.solver.dynamic_bumps()
    }
}
