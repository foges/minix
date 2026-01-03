//! Cut pool management for outer approximation.
//!
//! Manages the collection of cuts added during B&B, including:
//! - Cut storage and indexing
//! - Activity tracking
//! - Periodic cleanup of inactive cuts

use crate::master::LinearCut;

/// Status of a cut in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutStatus {
    /// Cut is active in the master problem.
    Active,

    /// Cut is in the pool but not in the master.
    Inactive,

    /// Cut has been permanently removed.
    Deleted,
}

/// A cut with pool metadata.
#[derive(Debug, Clone)]
pub struct PooledCut {
    /// The underlying linear cut.
    pub cut: LinearCut,

    /// Unique ID in the pool.
    pub id: usize,

    /// Current status.
    pub status: CutStatus,

    /// Number of times this cut was binding (dual > tol).
    pub times_binding: usize,

    /// Number of consecutive iterations where cut was slack.
    pub slack_count: usize,

    /// Iteration when cut was added.
    pub added_iter: usize,

    /// Last iteration when cut was binding.
    pub last_binding_iter: usize,
}

/// Cut pool settings.
#[derive(Debug, Clone)]
pub struct CutPoolSettings {
    /// Maximum cuts to keep in pool.
    pub max_cuts: usize,

    /// Remove cuts after this many consecutive slack iterations.
    pub max_slack_count: usize,

    /// How often to run cleanup (in iterations).
    pub cleanup_freq: usize,

    /// Minimum activity ratio to keep a cut.
    pub min_activity_ratio: f64,
}

impl Default for CutPoolSettings {
    fn default() -> Self {
        Self {
            max_cuts: 10000,
            max_slack_count: 50,
            cleanup_freq: 100,
            min_activity_ratio: 0.01,
        }
    }
}

/// Cut pool for managing generated cuts.
pub struct CutPool {
    /// All cuts in the pool.
    cuts: Vec<PooledCut>,

    /// Next cut ID.
    next_id: usize,

    /// Current iteration.
    iteration: usize,

    /// Settings.
    settings: CutPoolSettings,

    /// Statistics.
    stats: CutPoolStats,
}

/// Statistics for the cut pool.
#[derive(Debug, Default, Clone)]
pub struct CutPoolStats {
    /// Total cuts added.
    pub total_added: usize,

    /// Total cuts removed.
    pub total_removed: usize,

    /// Current active cuts.
    pub active_cuts: usize,

    /// Peak pool size.
    pub peak_size: usize,
}

impl CutPool {
    /// Create a new cut pool.
    pub fn new(settings: CutPoolSettings) -> Self {
        Self {
            cuts: Vec::new(),
            next_id: 0,
            iteration: 0,
            settings,
            stats: CutPoolStats::default(),
        }
    }

    /// Add a cut to the pool.
    ///
    /// Returns the cut ID and whether it's a duplicate.
    pub fn add(&mut self, cut: LinearCut) -> (usize, bool) {
        // Check for duplicates
        for pooled in &self.cuts {
            if pooled.status != CutStatus::Deleted && self.is_duplicate(&cut, &pooled.cut) {
                return (pooled.id, true);
            }
        }

        let id = self.next_id;
        self.next_id += 1;

        let pooled = PooledCut {
            cut,
            id,
            status: CutStatus::Active,
            times_binding: 0,
            slack_count: 0,
            added_iter: self.iteration,
            last_binding_iter: self.iteration,
        };

        self.cuts.push(pooled);
        self.stats.total_added += 1;
        self.stats.active_cuts += 1;
        self.stats.peak_size = self.stats.peak_size.max(self.cuts.len());

        (id, false)
    }

    /// Check if two cuts are duplicates.
    fn is_duplicate(&self, a: &LinearCut, b: &LinearCut) -> bool {
        // Check dimensions
        if a.coefs.len() != b.coefs.len() {
            return false;
        }

        // Check if cuts are parallel (within tolerance)
        let a_norm: f64 = a.coefs.iter().map(|x| x * x).sum::<f64>().sqrt();
        let b_norm: f64 = b.coefs.iter().map(|x| x * x).sum::<f64>().sqrt();

        if a_norm < 1e-10 || b_norm < 1e-10 {
            return a_norm < 1e-10 && b_norm < 1e-10;
        }

        // Compute dot product
        let dot: f64 = a.coefs.iter().zip(&b.coefs).map(|(ai, bi)| ai * bi).sum();
        let cos_angle = dot / (a_norm * b_norm);

        // Parallel if cos(angle) ≈ 1 and RHS similar
        if cos_angle.abs() > 0.9999 {
            let rhs_diff = (a.rhs / a_norm - b.rhs / b_norm).abs();
            return rhs_diff < 1e-8;
        }

        false
    }

    /// Update cut activity based on dual values.
    ///
    /// `dual_values` maps cut ID to its dual value in the master solution.
    pub fn update_activity(&mut self, dual_values: &[(usize, f64)]) {
        self.iteration += 1;

        // Build map of active duals
        let mut active_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for &(id, dual) in dual_values {
            if dual.abs() > 1e-8 {
                active_ids.insert(id);
            }
        }

        // Update each cut
        for pooled in &mut self.cuts {
            if pooled.status == CutStatus::Deleted {
                continue;
            }

            if active_ids.contains(&pooled.id) {
                pooled.times_binding += 1;
                pooled.slack_count = 0;
                pooled.last_binding_iter = self.iteration;
            } else if pooled.status == CutStatus::Active {
                pooled.slack_count += 1;
            }
        }

        // Periodic cleanup
        if self.iteration % self.settings.cleanup_freq == 0 {
            self.cleanup();
        }
    }

    /// Remove inactive cuts.
    fn cleanup(&mut self) {
        for pooled in &mut self.cuts {
            if pooled.status != CutStatus::Active {
                continue;
            }

            // Remove if slack too long
            if pooled.slack_count >= self.settings.max_slack_count {
                pooled.status = CutStatus::Inactive;
                self.stats.active_cuts -= 1;
                continue;
            }

            // Remove if activity ratio too low
            let age = self.iteration - pooled.added_iter + 1;
            let activity_ratio = pooled.times_binding as f64 / age as f64;

            if age > 10 && activity_ratio < self.settings.min_activity_ratio {
                pooled.status = CutStatus::Inactive;
                self.stats.active_cuts -= 1;
            }
        }

        // Compact if too many deleted/inactive cuts
        if self.cuts.len() > 2 * self.settings.max_cuts {
            self.compact();
        }
    }

    /// Remove deleted cuts from storage.
    fn compact(&mut self) {
        let removed = self.cuts.iter().filter(|c| c.status == CutStatus::Deleted).count();
        self.cuts.retain(|c| c.status != CutStatus::Deleted);
        self.stats.total_removed += removed;
    }

    /// Get active cuts.
    pub fn active_cuts(&self) -> impl Iterator<Item = &PooledCut> {
        self.cuts.iter().filter(|c| c.status == CutStatus::Active)
    }

    /// Get a cut by ID.
    pub fn get(&self, id: usize) -> Option<&PooledCut> {
        self.cuts.iter().find(|c| c.id == id)
    }

    /// Get a mutable cut by ID.
    pub fn get_mut(&mut self, id: usize) -> Option<&mut PooledCut> {
        self.cuts.iter_mut().find(|c| c.id == id)
    }

    /// Mark a cut as deleted.
    pub fn delete(&mut self, id: usize) {
        let mut was_active = false;
        if let Some(pooled) = self.cuts.iter_mut().find(|c| c.id == id) {
            was_active = pooled.status == CutStatus::Active;
            pooled.status = CutStatus::Deleted;
        }
        if was_active {
            self.stats.active_cuts -= 1;
        }
        self.stats.total_removed += 1;
    }

    /// Reactivate an inactive cut.
    pub fn activate(&mut self, id: usize) -> bool {
        let mut activated = false;
        if let Some(pooled) = self.cuts.iter_mut().find(|c| c.id == id) {
            if pooled.status == CutStatus::Inactive {
                pooled.status = CutStatus::Active;
                pooled.slack_count = 0;
                activated = true;
            }
        }
        if activated {
            self.stats.active_cuts += 1;
        }
        activated
    }

    /// Get pool statistics.
    pub fn stats(&self) -> &CutPoolStats {
        &self.stats
    }

    /// Number of cuts in pool (including inactive).
    pub fn len(&self) -> usize {
        self.cuts.len()
    }

    /// Check if pool is empty.
    pub fn is_empty(&self) -> bool {
        self.cuts.is_empty()
    }

    /// Number of active cuts.
    pub fn num_active(&self) -> usize {
        self.stats.active_cuts
    }

    /// Current iteration.
    pub fn iteration(&self) -> usize {
        self.iteration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::master::CutSource;

    fn make_cut(coeffs: Vec<f64>, rhs: f64) -> LinearCut {
        LinearCut::new(coeffs, rhs, CutSource::KStarCertificate { cone_idx: 0 })
    }

    #[test]
    fn test_pool_add_and_get() {
        let mut pool = CutPool::new(CutPoolSettings::default());

        let cut1 = make_cut(vec![1.0, 2.0], 3.0);
        let cut2 = make_cut(vec![4.0, 5.0], 6.0);

        let (id1, dup1) = pool.add(cut1);
        let (id2, dup2) = pool.add(cut2);

        assert!(!dup1);
        assert!(!dup2);
        assert_ne!(id1, id2);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.num_active(), 2);
    }

    #[test]
    fn test_duplicate_detection() {
        let mut pool = CutPool::new(CutPoolSettings::default());

        let cut1 = make_cut(vec![1.0, 2.0], 3.0);
        let cut2 = make_cut(vec![1.0, 2.0], 3.0); // Same cut
        let cut3 = make_cut(vec![2.0, 4.0], 6.0); // Parallel cut (same after normalization)

        let (id1, dup1) = pool.add(cut1);
        let (id2, dup2) = pool.add(cut2);
        let (id3, dup3) = pool.add(cut3);

        assert!(!dup1);
        assert!(dup2);
        assert!(dup3);
        assert_eq!(id1, id2);
        assert_eq!(id1, id3);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn test_activity_tracking() {
        let mut pool = CutPool::new(CutPoolSettings {
            max_slack_count: 3,
            cleanup_freq: 1, // Run cleanup every iteration
            ..Default::default()
        });

        let cut = make_cut(vec![1.0, 2.0], 3.0);
        let (id, _) = pool.add(cut);

        // Simulate iterations where cut is slack
        for _ in 0..3 {
            pool.update_activity(&[]);
        }

        // Cut should still be active (at threshold)
        assert_eq!(pool.get(id).unwrap().slack_count, 3);

        // One more slack iteration
        pool.update_activity(&[]);

        // Cut should be inactive now
        assert_eq!(pool.get(id).unwrap().status, CutStatus::Inactive);
    }

    #[test]
    fn test_cut_binding() {
        let mut pool = CutPool::new(CutPoolSettings::default());

        let cut = make_cut(vec![1.0, 2.0], 3.0);
        let (id, _) = pool.add(cut);

        // Cut is binding
        pool.update_activity(&[(id, 0.5)]);
        assert_eq!(pool.get(id).unwrap().times_binding, 1);
        assert_eq!(pool.get(id).unwrap().slack_count, 0);

        // Cut is slack
        pool.update_activity(&[]);
        assert_eq!(pool.get(id).unwrap().times_binding, 1);
        assert_eq!(pool.get(id).unwrap().slack_count, 1);

        // Cut is binding again (resets slack count)
        pool.update_activity(&[(id, 0.1)]);
        assert_eq!(pool.get(id).unwrap().times_binding, 2);
        assert_eq!(pool.get(id).unwrap().slack_count, 0);
    }
}
