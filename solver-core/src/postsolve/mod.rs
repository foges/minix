#[derive(Debug, Clone)]
pub struct PostsolveMap {
    orig_n: usize,
    shift: Vec<f64>,
    kept_indices: Vec<usize>,
    row_map: Option<RowMap>,
}

#[derive(Debug, Clone)]
pub struct RowMap {
    orig_m: usize,
    kept_rows: Vec<usize>,
    removed_rows: Vec<RemovedRow>,
}

#[derive(Debug, Clone)]
pub struct RemovedRow {
    pub row: usize,
    pub col: usize,
    pub val: f64,
    pub rhs: f64,
    pub kind: RemovedRowKind,
}

#[derive(Debug, Clone, Copy)]
pub enum RemovedRowKind {
    Zero,
    NonNeg,
}

impl PostsolveMap {
    pub fn identity(n: usize) -> Self {
        Self {
            orig_n: n,
            shift: vec![0.0; n],
            kept_indices: (0..n).collect(),
            row_map: None,
        }
    }

    pub fn new(orig_n: usize, shift: Vec<f64>, kept_indices: Vec<usize>) -> Self {
        Self {
            orig_n,
            shift,
            kept_indices,
            row_map: None,
        }
    }

    pub fn with_row_map(mut self, row_map: RowMap) -> Self {
        self.row_map = Some(row_map);
        self
    }

    pub fn orig_n(&self) -> usize {
        self.orig_n
    }

    /// Returns the expected size of the full s/z vectors after recovery.
    ///
    /// Given the reduced vector length (presolved constraints including bounds),
    /// computes the full output size needed for recover_s_into / recover_z_into.
    pub fn expected_sz_full_len(&self, reduced_len: usize) -> usize {
        let Some(row_map) = &self.row_map else {
            return reduced_len;
        };
        let kept_len = row_map.kept_rows.len();
        let bound_rows = reduced_len.saturating_sub(kept_len);
        row_map.orig_m + bound_rows
    }

    pub fn into_row_map(self) -> Option<RowMap> {
        self.row_map
    }

    pub fn recover_x(&self, x_reduced: &[f64]) -> Vec<f64> {
        let mut x = self.shift.clone();
        for (red_idx, &orig_idx) in self.kept_indices.iter().enumerate() {
            x[orig_idx] = x_reduced[red_idx] + self.shift[orig_idx];
        }
        x
    }

    pub fn recover_x_into(&self, x_reduced: &[f64], out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.orig_n);
        out.copy_from_slice(&self.shift);
        for (red_idx, &orig_idx) in self.kept_indices.iter().enumerate() {
            out[orig_idx] = x_reduced[red_idx] + self.shift[orig_idx];
        }
    }

    pub fn reduce_x(&self, x_full: &[f64]) -> Vec<f64> {
        let mut x_reduced = Vec::with_capacity(self.kept_indices.len());
        for &orig_idx in &self.kept_indices {
            x_reduced.push(x_full[orig_idx] - self.shift[orig_idx]);
        }
        x_reduced
    }

    pub fn reduce_s(&self, s_full: &[f64], target_m: usize) -> Vec<f64> {
        let Some(row_map) = &self.row_map else {
            return if s_full.len() == target_m {
                s_full.to_vec()
            } else {
                Vec::new()
            };
        };

        let kept_len = row_map.kept_rows.len();
        if target_m < kept_len {
            return Vec::new();
        }
        let bound_rows = target_m - kept_len;
        let expected_full = row_map.orig_m + bound_rows;

        if s_full.len() == target_m {
            return s_full.to_vec();
        }
        if s_full.len() != expected_full {
            return Vec::new();
        }

        let mut s_reduced = Vec::with_capacity(target_m);
        for &orig_row in &row_map.kept_rows {
            s_reduced.push(s_full[orig_row]);
        }
        for idx in 0..bound_rows {
            s_reduced.push(s_full[row_map.orig_m + idx]);
        }
        s_reduced
    }

    pub fn reduce_z(&self, z_full: &[f64], target_m: usize) -> Vec<f64> {
        let Some(row_map) = &self.row_map else {
            return if z_full.len() == target_m {
                z_full.to_vec()
            } else {
                Vec::new()
            };
        };

        let kept_len = row_map.kept_rows.len();
        if target_m < kept_len {
            return Vec::new();
        }
        let bound_rows = target_m - kept_len;
        let expected_full = row_map.orig_m + bound_rows;

        if z_full.len() == target_m {
            return z_full.to_vec();
        }
        if z_full.len() != expected_full {
            return Vec::new();
        }

        let mut z_reduced = Vec::with_capacity(target_m);
        for &orig_row in &row_map.kept_rows {
            z_reduced.push(z_full[orig_row]);
        }
        for idx in 0..bound_rows {
            z_reduced.push(z_full[row_map.orig_m + idx]);
        }
        z_reduced
    }

    pub fn recover_s(&self, s_reduced: &[f64], x_full: &[f64]) -> Vec<f64> {
        let Some(row_map) = &self.row_map else {
            return s_reduced.to_vec();
        };

        let kept_len = row_map.kept_rows.len();
        let bound_rows = s_reduced.len().saturating_sub(kept_len);
        let (s_base, s_bounds) = s_reduced.split_at(kept_len);

        let mut s_full = vec![0.0; row_map.orig_m + bound_rows];
        for (red_idx, &orig_row) in row_map.kept_rows.iter().enumerate() {
            s_full[orig_row] = s_base[red_idx];
        }
        for removed in &row_map.removed_rows {
            s_full[removed.row] = match removed.kind {
                RemovedRowKind::Zero => 0.0,
                RemovedRowKind::NonNeg => removed.rhs - removed.val * x_full[removed.col],
            };
        }
        for (idx, &val) in s_bounds.iter().enumerate() {
            s_full[row_map.orig_m + idx] = val;
        }

        s_full
    }

    pub fn recover_s_into(&self, s_reduced: &[f64], x_full: &[f64], out: &mut [f64]) {
        let Some(row_map) = &self.row_map else {
            debug_assert_eq!(out.len(), s_reduced.len());
            out.copy_from_slice(s_reduced);
            return;
        };

        let kept_len = row_map.kept_rows.len();
        let bound_rows = s_reduced.len().saturating_sub(kept_len);
        let (s_base, s_bounds) = s_reduced.split_at(kept_len);
        debug_assert_eq!(out.len(), row_map.orig_m + bound_rows);

        out.fill(0.0);
        for (red_idx, &orig_row) in row_map.kept_rows.iter().enumerate() {
            out[orig_row] = s_base[red_idx];
        }
        for removed in &row_map.removed_rows {
            out[removed.row] = match removed.kind {
                RemovedRowKind::Zero => 0.0,
                RemovedRowKind::NonNeg => removed.rhs - removed.val * x_full[removed.col],
            };
        }
        for (idx, &val) in s_bounds.iter().enumerate() {
            out[row_map.orig_m + idx] = val;
        }
    }

    pub fn recover_z(&self, z_reduced: &[f64]) -> Vec<f64> {
        let Some(row_map) = &self.row_map else {
            return z_reduced.to_vec();
        };

        let kept_len = row_map.kept_rows.len();
        let bound_rows = z_reduced.len().saturating_sub(kept_len);
        let (z_base, z_bounds) = z_reduced.split_at(kept_len);

        let mut z_full = vec![0.0; row_map.orig_m + bound_rows];
        for (red_idx, &orig_row) in row_map.kept_rows.iter().enumerate() {
            z_full[orig_row] = z_base[red_idx];
        }
        // Removed rows default to zero duals.
        for (idx, &val) in z_bounds.iter().enumerate() {
            z_full[row_map.orig_m + idx] = val;
        }

        z_full
    }

    pub fn recover_z_into(&self, z_reduced: &[f64], out: &mut [f64]) {
        let Some(row_map) = &self.row_map else {
            debug_assert_eq!(out.len(), z_reduced.len());
            out.copy_from_slice(z_reduced);
            return;
        };

        let kept_len = row_map.kept_rows.len();
        let bound_rows = z_reduced.len().saturating_sub(kept_len);
        let (z_base, z_bounds) = z_reduced.split_at(kept_len);
        debug_assert_eq!(out.len(), row_map.orig_m + bound_rows);

        out.fill(0.0);
        for (red_idx, &orig_row) in row_map.kept_rows.iter().enumerate() {
            out[orig_row] = z_base[red_idx];
        }
        for (idx, &val) in z_bounds.iter().enumerate() {
            out[row_map.orig_m + idx] = val;
        }
    }
}

impl RowMap {
    pub fn new(orig_m: usize, kept_rows: Vec<usize>, removed_rows: Vec<RemovedRow>) -> Self {
        Self {
            orig_m,
            kept_rows,
            removed_rows,
        }
    }
}
