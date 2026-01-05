use sprs::CsMat;

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
