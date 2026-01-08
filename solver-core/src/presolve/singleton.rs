use sprs::CsMat;
use crate::problem::ConeSpec;

#[derive(Debug, Clone)]
pub struct SingletonRow {
    pub row: usize,
    pub col: usize,
    pub val: f64,
}

#[derive(Debug, Clone)]
pub struct SingletonPartition {
    pub singleton_rows: Vec<SingletonRow>,
    pub non_singleton_rows: Vec<usize>,
}

/// Check if a row is eligible for singleton elimination based on its cone type.
/// Only separable 1D cones (Zero, NonNeg) are safe for row-wise elimination.
fn row_is_eligible_for_singleton_elim(row: usize, cones: &[ConeSpec]) -> bool {
    let mut offset = 0usize;
    for cone in cones {
        let dim = cone.dim();
        if row < offset + dim {
            // This row belongs to this cone
            return match cone {
                ConeSpec::Zero { dim } if *dim == 1 => true,
                ConeSpec::NonNeg { dim } if *dim == 1 => true,
                _ => false, // SOC, Exp, PSD, or multi-dimensional Zero/NonNeg are NOT safe
            };
        }
        offset += dim;
    }
    false // Row not found in any cone (shouldn't happen)
}

/// Detect singleton rows, filtering out rows that belong to multi-dimensional cones.
/// Only returns singletons from separable 1D cones (Zero, NonNeg).
pub fn detect_singleton_rows_cone_aware(a: &CsMat<f64>, cones: &[ConeSpec]) -> SingletonPartition {
    let m = a.rows();
    let n = a.cols();

    let mut counts = vec![0u8; m];
    let mut col_idx = vec![usize::MAX; m];
    let mut vals = vec![0.0; m];

    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            for (row, &val) in col_view.iter() {
                if counts[row] == 0 {
                    counts[row] = 1;
                    col_idx[row] = col;
                    vals[row] = val;
                } else {
                    counts[row] = 2;
                }
            }
        }
    }

    let mut singleton_rows = Vec::new();
    let mut non_singleton_rows = Vec::new();
    for row in 0..m {
        if counts[row] == 1 {
            // Only add if row is eligible for singleton elimination (1D separable cone)
            if row_is_eligible_for_singleton_elim(row, cones) {
                singleton_rows.push(SingletonRow {
                    row,
                    col: col_idx[row],
                    val: vals[row],
                });
            } else {
                non_singleton_rows.push(row);
            }
        } else {
            non_singleton_rows.push(row);
        }
    }

    SingletonPartition {
        singleton_rows,
        non_singleton_rows,
    }
}

/// Legacy version without cone awareness (deprecated, kept for compatibility).
pub fn detect_singleton_rows(a: &CsMat<f64>) -> SingletonPartition {
    let m = a.rows();
    let n = a.cols();

    let mut counts = vec![0u8; m];
    let mut col_idx = vec![usize::MAX; m];
    let mut vals = vec![0.0; m];

    for col in 0..n {
        if let Some(col_view) = a.outer_view(col) {
            for (row, &val) in col_view.iter() {
                if counts[row] == 0 {
                    counts[row] = 1;
                    col_idx[row] = col;
                    vals[row] = val;
                } else {
                    counts[row] = 2;
                }
            }
        }
    }

    let mut singleton_rows = Vec::new();
    let mut non_singleton_rows = Vec::new();
    for row in 0..m {
        if counts[row] == 1 {
            singleton_rows.push(SingletonRow {
                row,
                col: col_idx[row],
                val: vals[row],
            });
        } else {
            non_singleton_rows.push(row);
        }
    }

    SingletonPartition {
        singleton_rows,
        non_singleton_rows,
    }
}
