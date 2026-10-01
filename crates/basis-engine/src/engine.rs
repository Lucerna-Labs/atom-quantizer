use crate::{Field, LocalRequest, Request};
use basis_primitives::Primitive::{self, *};

const ORDERS: [[Primitive; 7]; 2] = [
    [
        Unitary,
        Binding,
        Depression,
        Information,
        Oscillator,
        Rotor,
        Population,
    ],
    [
        Unitary,
        Rotor,
        Population,
        Binding,
        Depression,
        Information,
        Oscillator,
    ],
];

pub(super) fn evolve(request: Request) -> Result<Field, String> {
    if request.disabled > 255 {
        return Err("unknown primitive mask".into());
    }
    let stages = ORDERS.get(request.order).ok_or("unknown composition")?;
    let active = |p: &&Primitive| (**p as u16) & request.disabled == 0;
    let field = [Interference]
        .iter()
        .filter(active)
        .try_fold(Field::new(request.signal, request.seed)?, |f, p| p.apply(f))?;
    (0..16).try_fold(field, |field, _| {
        stages
            .iter()
            .filter(active)
            .try_fold(field, |f, p| p.apply(f))
    })
}

pub(super) fn evolve_local(request: LocalRequest) -> Result<Field, String> {
    let input = request.input;
    if input.disabled & !510 != 0 {
        return Err("unknown local primitive mask".into());
    }
    let stages = ORDERS.get(input.order).ok_or("unknown local composition")?;
    let active = |p: &&Primitive| (**p as u16) & input.disabled == 0;
    let field = [Phase].iter().filter(active).try_fold(
        Field::coupled(input.signal, input.seed, request.coupling)?,
        |f, p| p.apply(f),
    )?;
    (0..16).try_fold(field, |field, _| {
        stages
            .iter()
            .filter(active)
            .try_fold(field, |f, p| p.apply(f))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WIDTH;

    #[test]
    fn every_precommitted_control_conserves_transport_and_repeats_exactly() {
        for order in 0..2 {
            for disabled in [0, 1, 2, 4, 8, 16, 32, 64, 128, 3, 192, 252, 255] {
                for seed in 1..=3 {
                    let request = || Request {
                        signal: std::array::from_fn(|i| (i as f64 + 0.3).cos()),
                        seed,
                        order,
                        disabled,
                    };
                    let result = crate::run(request()).unwrap();
                    assert!(result.diagnostics()[0] < 1e-10);
                    assert!((result.diagnostics()[1] - 1.0).abs() < 1e-10);
                    assert_eq!(result.matrix(), crate::run(request()).unwrap().matrix());
                }
            }
        }
    }

    #[test]
    fn identity_and_invalid_compositions_are_explicit() {
        let request = |order, disabled| Request {
            signal: [1.0; WIDTH],
            seed: 1,
            order,
            disabled,
        };
        let result = crate::run(request(0, 255)).unwrap();
        for i in 0..WIDTH {
            for j in 0..WIDTH {
                assert_eq!(result.matrix()[i][j], f64::from(i == j));
            }
        }
        assert!(crate::run(request(2, 0)).is_err());
        assert!(crate::run(request(0, 256)).is_err());
    }

    #[test]
    fn local_orders_and_controls_preserve_invariants_and_repetition() {
        for order in 0..2 {
            for disabled in [0, 256, 2, 4, 8, 16, 32, 64, 128, 258, 192, 252, 510] {
                for seed in 1..=3 {
                    let request = || LocalRequest {
                        input: Request {
                            signal: [1.0; WIDTH],
                            order,
                            disabled,
                            seed,
                        },
                        coupling: std::array::from_fn(|i| {
                            std::array::from_fn(|j| ((i + j + 1) as f64 * 0.37).cos())
                        }),
                    };
                    let field = crate::run_local(request()).unwrap();
                    assert!(field.diagnostics()[0] < 1e-10);
                    assert!((field.diagnostics()[1] - 1.0).abs() < 1e-10);
                    assert_eq!(
                        field.matrix(),
                        crate::run_local(request()).unwrap().matrix()
                    );
                }
            }
        }
    }
}
