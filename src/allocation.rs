//! Global byte-budget allocation over accepted tensor candidates.
//!
//! Lagrangian search followed by marginal-gain filling is deterministic and
//! budget-feasible. It is not claimed to solve the discrete knapsack optimally.

#[derive(Clone, Copy, Debug)]
pub struct Choice {
    pub bytes: u64,
    pub distortion: f64,
}

#[derive(Debug)]
pub struct Allocation {
    pub choices: Vec<usize>,
    pub bytes: u64,
    pub distortion: f64,
    pub minimum_bytes: u64,
}

fn totals(groups: &[Vec<Choice>], chosen: &[usize]) -> Result<(u64, f64), String> {
    let (mut bytes, mut loss) = (0u64, 0.0);
    for (group, &i) in groups.iter().zip(chosen) {
        bytes = bytes
            .checked_add(group[i].bytes)
            .ok_or("allocation byte count overflow")?;
        loss += group[i].distortion;
    }
    if !loss.is_finite() {
        return Err("allocation distortion overflow".into());
    }
    Ok((bytes, loss))
}

pub fn allocate(groups: &[Vec<Choice>], budget: u64) -> Result<Allocation, String> {
    for group in groups {
        if group.is_empty()
            || group
                .iter()
                .any(|c| !c.distortion.is_finite() || c.distortion < 0.0)
        {
            return Err("every tensor needs finite, nonnegative accepted candidates".into());
        }
    }
    let cheapest: Vec<_> = groups
        .iter()
        .map(|g| {
            g.iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.bytes
                        .cmp(&b.1.bytes)
                        .then(a.1.distortion.total_cmp(&b.1.distortion))
                })
                .unwrap()
                .0
        })
        .collect();
    let (minimum_bytes, _) = totals(groups, &cheapest)?;
    if minimum_bytes > budget {
        return Err(format!("budget infeasible: accepted records require at least {minimum_bytes} bytes; budget is {budget}"));
    }
    let choose = |lambda: f64| -> Vec<usize> {
        groups
            .iter()
            .map(|g| {
                g.iter()
                    .enumerate()
                    .min_by(|a, b| {
                        let av = a.1.distortion + lambda * a.1.bytes as f64;
                        let bv = b.1.distortion + lambda * b.1.bytes as f64;
                        av.total_cmp(&bv).then(a.1.bytes.cmp(&b.1.bytes))
                    })
                    .unwrap()
                    .0
            })
            .collect()
    };
    let best_quality = choose(0.0);
    let mut chosen = if totals(groups, &best_quality)?.0 <= budget {
        best_quality
    } else {
        let (mut low, mut high) = (0.0, 1.0);
        while totals(groups, &choose(high))?.0 > budget {
            high *= 2.0;
            if !high.is_finite() {
                return Err("cannot bracket allocation multiplier".into());
            }
        }
        let mut feasible = cheapest;
        for _ in 0..80 {
            let mid = low + (high - low) * 0.5;
            let candidate = choose(mid);
            if totals(groups, &candidate)?.0 <= budget {
                feasible = candidate;
                high = mid;
            } else {
                low = mid;
            }
        }
        feasible
    };
    let (mut bytes, _) = totals(groups, &chosen)?;
    loop {
        let mut upgrade: Option<(usize, usize, f64)> = None;
        for (gi, group) in groups.iter().enumerate() {
            let current = group[chosen[gi]];
            for (ci, next) in group.iter().enumerate() {
                if next.bytes <= current.bytes || next.distortion >= current.distortion {
                    continue;
                }
                let delta = next.bytes - current.bytes;
                if delta > budget - bytes {
                    continue;
                }
                let benefit = (current.distortion - next.distortion) / delta as f64;
                if upgrade.is_none_or(|(_, _, old)| benefit > old) {
                    upgrade = Some((gi, ci, benefit));
                }
            }
        }
        let Some((g, c, _)) = upgrade else {
            break;
        };
        bytes += groups[g][c].bytes - groups[g][chosen[g]].bytes;
        chosen[g] = c;
    }
    let (bytes, distortion) = totals(groups, &chosen)?;
    debug_assert!(bytes <= budget);
    Ok(Allocation {
        choices: chosen,
        bytes,
        distortion,
        minimum_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_follow_functional_benefit_across_tensors() {
        let groups = vec![
            vec![
                Choice {
                    bytes: 10,
                    distortion: 10.0,
                },
                Choice {
                    bytes: 20,
                    distortion: 1.0,
                },
            ],
            vec![
                Choice {
                    bytes: 10,
                    distortion: 2.0,
                },
                Choice {
                    bytes: 20,
                    distortion: 1.0,
                },
            ],
        ];
        let a = allocate(&groups, 30).unwrap();
        assert_eq!(a.choices, vec![1, 0]);
        assert_eq!(a.bytes, 30);
        assert_eq!(a.distortion, 3.0);
        assert!(allocate(&groups, 19).is_err());
    }

    #[test]
    fn irregular_budgets_never_silently_overspend() {
        let groups: Vec<_> = (0..12)
            .map(|i| {
                vec![
                    Choice {
                        bytes: 7 + i,
                        distortion: 1.0,
                    },
                    Choice {
                        bytes: 19 + i * 2,
                        distortion: 0.2 + i as f64 * 0.01,
                    },
                    Choice {
                        bytes: 47 + i * 3,
                        distortion: 0.0,
                    },
                ]
            })
            .collect();
        let minimum: u64 = groups.iter().map(|g| g[0].bytes).sum();
        for budget in minimum..minimum + 300 {
            let a = allocate(&groups, budget).unwrap();
            assert!(a.bytes <= budget);
            assert_eq!(
                a.bytes,
                groups
                    .iter()
                    .zip(a.choices)
                    .map(|(g, i)| g[i].bytes)
                    .sum::<u64>()
            );
        }
    }
}
