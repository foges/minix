//! SOC tangent cut generation.
//!
//! Generates outer approximation cuts for second-order cone constraints.
//!
//! For an SOC constraint (t, x) ∈ SOC (i.e., t >= ||x||), we can generate
//! tangent hyperplane cuts at any point on the cone boundary.

use solver_core::ConeSpec;

use crate::master::{CutSource, LinearCut};
use crate::model::MipProblem;

/// SOC tangent cut generator.
pub struct SocTangentGenerator {
    /// Settings.
    settings: SocTangentSettings,

    /// Statistics.
    stats: SocTangentStats,
}

/// Settings for SOC tangent cut generation.
#[derive(Debug, Clone)]
pub struct SocTangentSettings {
    /// Minimum norm of x to generate tangent cut (avoid singularity at apex).
    pub min_norm: f64,

    /// Maximum cuts per SOC cone per round.
    pub max_cuts_per_cone: usize,

    /// Minimum violation for generated cuts.
    pub min_violation: f64,
}

impl Default for SocTangentSettings {
    fn default() -> Self {
        Self {
            min_norm: 1e-8,
            max_cuts_per_cone: 3,
            min_violation: 1e-8,
        }
    }
}

/// Statistics for SOC tangent generation.
#[derive(Debug, Default, Clone)]
pub struct SocTangentStats {
    /// Total cuts generated.
    pub cuts_generated: usize,

    /// Cuts rejected (at apex, not violated, etc.).
    pub cuts_rejected: usize,
}

impl SocTangentGenerator {
    /// Create a new SOC tangent generator.
    pub fn new(settings: SocTangentSettings) -> Self {
        Self {
            settings,
            stats: SocTangentStats::default(),
        }
    }

    /// Generate tangent cuts at violated SOC constraints.
    ///
    /// For each SOC constraint where (t, x) violates t >= ||x||, generates
    /// a tangent cut at the boundary point.
    ///
    /// # Arguments
    ///
    /// * `prob` - MIP problem
    /// * `s` - Current slack vector (s = b - Ax)
    ///
    /// # Returns
    ///
    /// Vector of tangent cuts for violated SOC constraints.
    pub fn generate(
        &mut self,
        prob: &MipProblem,
        s: &[f64],
    ) -> Vec<LinearCut> {
        let mut cuts = Vec::new();
        let mut offset = 0;

        for (cone_idx, cone) in prob.conic.cones.iter().enumerate() {
            match cone {
                ConeSpec::Soc { dim } => {
                    let dim = *dim;
                    if dim < 2 {
                        offset += dim;
                        continue;
                    }

                    let s_block = &s[offset..offset + dim];
                    let t = s_block[0];
                    let x = &s_block[1..];

                    // Check if constraint is violated: t < ||x||
                    let x_norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();

                    if t < x_norm - self.settings.min_violation {
                        // Generate tangent cut
                        if let Some(cut) = self.generate_tangent_cut(
                            prob,
                            cone_idx,
                            offset,
                            dim,
                            t,
                            x,
                            x_norm,
                        ) {
                            cuts.push(cut);
                            self.stats.cuts_generated += 1;

                            if cuts.len() >= self.settings.max_cuts_per_cone {
                                break;
                            }
                        } else {
                            self.stats.cuts_rejected += 1;
                        }
                    }

                    offset += dim;
                }
                _ => {
                    offset += cone.dim();
                }
            }
        }

        cuts
    }

    /// Generate a single tangent cut at a violated SOC point.
    ///
    /// For SOC constraint: s[0] >= ||(s[1], ..., s[dim-1])||
    /// where s = b - Ax
    ///
    /// At boundary point (t*, x*) with ||x*|| = t*, the tangent hyperplane is:
    ///   t >= (x*)^T x / ||x*||
    ///
    /// Substituting s = b - Ax and rearranging:
    ///   -A[0,:] x + sum_i (x*[i]/||x*||) A[i,:] x <= -b[0] + sum_i (x*[i]/||x*||) b[i]
    fn generate_tangent_cut(
        &self,
        prob: &MipProblem,
        cone_idx: usize,
        offset: usize,
        dim: usize,
        t: f64,
        x: &[f64],
        x_norm: f64,
    ) -> Option<LinearCut> {
        // Avoid singularity at apex
        if x_norm < self.settings.min_norm {
            return None;
        }

        let n = prob.num_vars();
        let mut a_cut = vec![0.0; n];

        // Compute normalized direction
        let x_hat: Vec<f64> = x.iter().map(|xi| xi / x_norm).collect();

        // Build cut coefficients: a_cut = -A[offset,:] + sum_i x_hat[i] * A[offset+1+i,:]
        for (col_idx, col) in prob.conic.A.outer_iterator().enumerate() {
            for (row_idx, &val) in col.iter() {
                if row_idx == offset {
                    // -A[0,:] (negated because s = b - Ax, so A x + s = b)
                    a_cut[col_idx] -= val;
                } else if row_idx > offset && row_idx < offset + dim {
                    let local_idx = row_idx - offset - 1;
                    a_cut[col_idx] += x_hat[local_idx] * val;
                }
            }
        }

        // Compute RHS: -b[offset] + sum_i x_hat[i] * b[offset+1+i]
        let mut rhs = -prob.conic.b[offset];
        for (i, &xi_hat) in x_hat.iter().enumerate() {
            rhs += xi_hat * prob.conic.b[offset + 1 + i];
        }

        let mut cut = LinearCut::new(
            a_cut,
            rhs,
            CutSource::SocTangent { cone_idx },
        );
        cut.normalize();

        if cut.is_valid() {
            Some(cut)
        } else {
            None
        }
    }

    /// Generate multiple tangent cuts from different directions.
    ///
    /// Generates cuts not just at the current point, but also at nearby
    /// points to better approximate the cone.
    pub fn generate_multi_tangent(
        &mut self,
        prob: &MipProblem,
        s: &[f64],
        num_directions: usize,
    ) -> Vec<LinearCut> {
        let mut cuts = self.generate(prob, s);

        if num_directions <= 1 {
            return cuts;
        }

        // Generate additional cuts at perturbed directions
        // This helps with the initial approximation
        let mut offset = 0;
        for (cone_idx, cone) in prob.conic.cones.iter().enumerate() {
            if let ConeSpec::Soc { dim } = cone {
                let dim = *dim;
                if dim < 3 {
                    offset += dim;
                    continue;
                }

                let s_block = &s[offset..offset + dim];
                let t = s_block[0];
                let x = &s_block[1..];
                let x_norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();

                // Only generate extra cuts if significantly violated
                if t < x_norm - 10.0 * self.settings.min_violation {
                    // Generate cuts at coordinate directions
                    for dir in 0..(dim - 1).min(num_directions - 1) {
                        let mut x_perturbed = x.to_vec();
                        // Perturb in coordinate direction
                        let scale = x_norm / (dim as f64).sqrt();
                        x_perturbed[dir] += scale * 0.1;
                        let perturbed_norm: f64 =
                            x_perturbed.iter().map(|xi| xi * xi).sum::<f64>().sqrt();

                        if let Some(cut) = self.generate_tangent_cut(
                            prob,
                            cone_idx,
                            offset,
                            dim,
                            t,
                            &x_perturbed,
                            perturbed_norm,
                        ) {
                            cuts.push(cut);
                            self.stats.cuts_generated += 1;
                        }
                    }
                }

                offset += dim;
            } else {
                offset += cone.dim();
            }
        }

        cuts
    }

    /// Get generation statistics.
    pub fn stats(&self) -> &SocTangentStats {
        &self.stats
    }
}

/// Helper: Compute violation of SOC constraint.
///
/// Returns positive value if violated (t < ||x||).
pub fn soc_violation(t: f64, x: &[f64]) -> f64 {
    let x_norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
    x_norm - t
}

/// Helper: Project point onto SOC boundary.
///
/// Given (t, x), returns the closest point on the cone boundary.
pub fn project_to_soc_boundary(t: f64, x: &[f64]) -> (f64, Vec<f64>) {
    let x_norm: f64 = x.iter().map(|xi| xi * xi).sum::<f64>().sqrt();

    if x_norm < 1e-10 {
        // At apex, project to apex
        return (0.0, vec![0.0; x.len()]);
    }

    // On boundary: t* = ||x*||
    // Project (t, x) -> (t*, x*) where ||x*|| = t*
    // The projection is: t* = (t + ||x||) / 2, x* = x * t* / ||x||
    let t_star = (t + x_norm) / 2.0;
    let scale = t_star / x_norm;
    let x_star: Vec<f64> = x.iter().map(|xi| xi * scale).collect();

    (t_star, x_star)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_soc_violation() {
        // On boundary: t = ||x||
        assert!((soc_violation(1.0, &[1.0])).abs() < 1e-10);
        assert!((soc_violation(5.0f64.sqrt(), &[1.0, 2.0])).abs() < 1e-10);

        // Interior: t > ||x||
        assert!(soc_violation(2.0, &[1.0]) < 0.0);

        // Violated: t < ||x||
        assert!(soc_violation(0.5, &[1.0]) > 0.0);
    }

    #[test]
    fn test_project_to_boundary() {
        // Interior point
        let (t_star, x_star) = project_to_soc_boundary(3.0, &[1.0]);
        let x_norm: f64 = x_star.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        assert!((t_star - x_norm).abs() < 1e-10);

        // Exterior point
        let (t_star, x_star) = project_to_soc_boundary(0.5, &[2.0]);
        let x_norm: f64 = x_star.iter().map(|xi| xi * xi).sum::<f64>().sqrt();
        assert!((t_star - x_norm).abs() < 1e-10);
    }
}
