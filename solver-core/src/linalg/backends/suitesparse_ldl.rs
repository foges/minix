use sprs_suitesparse_ldl::{LdlNumeric, LdlSymbolic};

use crate::linalg::backend::{BackendError, KktBackend};
use crate::linalg::sparse::SparseCsc;

pub struct SuiteSparseLdlBackend {
    n: usize,
    static_reg: f64,
    symbolic: Option<LdlSymbolic>,
    numeric: Option<LdlNumeric>,
}

impl SuiteSparseLdlBackend {
    fn with_static_reg(&self, mat: &SparseCsc) -> SparseCsc {
        if self.static_reg == 0.0 {
            return mat.clone();
        }

        let mut mat_reg = mat.clone();

        // First, collect diagonal positions from immutable view
        let indptr = mat_reg.indptr();
        let col_ptr = indptr.raw_storage();
        let row_idx = mat_reg.indices();

        let mut diag_positions = Vec::with_capacity(self.n);
        for col in 0..self.n {
            let start = col_ptr[col];
            let end = col_ptr[col + 1];
            for idx in start..end {
                if row_idx[idx] == col {
                    diag_positions.push(idx);
                    break;
                }
            }
        }

        // Now mutate data
        let data = mat_reg.data_mut();
        for &idx in &diag_positions {
            data[idx] += self.static_reg;
        }

        mat_reg
    }
}

impl KktBackend for SuiteSparseLdlBackend {
    type Factorization = ();

    fn new(n: usize, static_reg: f64, _dynamic_reg_min_pivot: f64) -> Self {
        Self {
            n,
            static_reg,
            symbolic: None,
            numeric: None,
        }
    }

    fn set_static_reg(&mut self, static_reg: f64) -> Result<(), BackendError> {
        if !static_reg.is_finite() || static_reg < 0.0 {
            return Err(BackendError::Message(format!(
                "invalid static_reg {}",
                static_reg
            )));
        }
        self.static_reg = static_reg;
        Ok(())
    }

    fn static_reg(&self) -> f64 {
        self.static_reg
    }

    fn symbolic_factorization(&mut self, kkt: &SparseCsc) -> Result<(), BackendError> {
        let kkt_reg = self.with_static_reg(kkt);
        self.symbolic = Some(LdlSymbolic::new(kkt_reg.view()));
        self.numeric = None;
        Ok(())
    }

    fn numeric_factorization(&mut self, kkt: &SparseCsc) -> Result<Self::Factorization, BackendError> {
        let kkt_reg = self.with_static_reg(kkt);

        if self.symbolic.is_none() {
            self.symbolic = Some(LdlSymbolic::new(kkt_reg.view()));
        }

        if let Some(numeric) = self.numeric.as_mut() {
            numeric
                .update(kkt_reg.view())
                .map_err(|e| BackendError::Message(format!("SuiteSparse LDL update failed: {}", e)))?;
        } else {
            let symbolic = self
                .symbolic
                .as_ref()
                .expect("symbolic factorization missing")
                .clone();
            let numeric = symbolic
                .factor(kkt_reg.view())
                .map_err(|e| BackendError::Message(format!("SuiteSparse LDL factor failed: {}", e)))?;
            self.numeric = Some(numeric);
        }

        Ok(())
    }

    fn solve(&self, _factor: &Self::Factorization, rhs: &[f64], sol: &mut [f64]) {
        if let Some(numeric) = self.numeric.as_ref() {
            let rhs_vec: Vec<f64> = rhs.to_vec();
            let x = numeric.solve(&rhs_vec);
            sol.copy_from_slice(&x);
        } else {
            sol.copy_from_slice(rhs);
        }
    }

    fn dynamic_bumps(&self) -> u64 {
        0
    }
}
