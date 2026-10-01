//! Intrinsic B01 mechanics. No quantizer, score, target basis or model identity.
use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2, TAU};

pub const WIDTH: usize = 32;
pub const DT: f64 = 0.125;
pub type Matrix = [[f64; WIDTH]; WIDTH];

#[derive(Clone, Copy, Debug)]
#[repr(u16)]
pub enum Primitive {
    Interference = 1,
    Unitary = 2,
    Binding = 4,
    Depression = 8,
    Information = 16,
    Oscillator = 32,
    Rotor = 64,
    Population = 128,
    Phase = 256,
}

#[derive(Clone)]
pub struct Field {
    amplitudes: [f64; WIDTH],
    transport: Matrix,
    bound: [f64; WIDTH],
    resource: [f64; WIDTH],
    position: [f64; WIDTH],
    velocity: [f64; WIDTH],
    phase: [f64; WIDTH],
    momentum: [f64; WIDTH],
    population: [f64; WIDTH],
    coupling: Matrix,
}

impl Field {
    pub fn new(signal: [f64; WIDTH], seed: u32) -> Result<Self, String> {
        let energy: f64 = signal.iter().map(|x| x * x).sum();
        if !energy.is_finite() || energy <= 0.0 || signal.iter().any(|x| !x.is_finite()) {
            return Err("stimulus must have positive finite energy".into());
        }
        let phase = std::array::from_fn(|i| (((i + 1) as f64 + f64::from(seed)) * SQRT_2).fract());
        Ok(Self {
            amplitudes: signal.map(|x| x / energy.sqrt()),
            transport: std::array::from_fn(|i| std::array::from_fn(|j| f64::from(i == j))),
            bound: [0.0; WIDTH],
            resource: [1.0; WIDTH],
            position: [0.0; WIDTH],
            velocity: [0.0; WIDTH],
            phase: phase.map(|x| x * TAU),
            momentum: [0.0; WIDTH],
            population: phase,
            coupling: [[1.0; WIDTH]; WIDTH],
        })
    }

    pub fn coupled(signal: [f64; WIDTH], seed: u32, coupling: Matrix) -> Result<Self, String> {
        for (i, row) in coupling.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                if !value.is_finite()
                    || value.abs() > 1.0 + 1e-12
                    || (value - coupling[j][i]).abs() > 1e-12
                {
                    return Err("coupling must be finite, symmetric and bounded by one".into());
                }
            }
        }
        let mut field = Self::new(signal, seed)?;
        field.coupling = coupling;
        Ok(field)
    }

    pub fn matrix(&self) -> &Matrix {
        &self.transport
    }

    pub fn resources(&self) -> &[f64; WIDTH] {
        &self.resource
    }

    pub fn diagnostics(&self) -> [f64; 6] {
        let energy: f64 = self.amplitudes.iter().map(|x| x * x).sum();
        let orthogonality = (0..WIDTH)
            .flat_map(|i| {
                (0..WIDTH).map(move |j| {
                    (self.transport[i]
                        .iter()
                        .zip(self.transport[j])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        - f64::from(i == j))
                    .abs()
                })
            })
            .fold(0.0, f64::max);
        [
            orthogonality,
            energy,
            self.bound.iter().copied().fold(1.0, f64::min),
            self.bound.iter().copied().fold(0.0, f64::max),
            self.resource.iter().copied().fold(1.0, f64::min),
            self.resource.iter().copied().fold(0.0, f64::max),
        ]
    }

    pub fn validate(self) -> Result<Self, String> {
        let scalars = [
            &self.amplitudes,
            &self.bound,
            &self.resource,
            &self.position,
            &self.velocity,
            &self.phase,
            &self.momentum,
            &self.population,
        ];
        if scalars
            .iter()
            .flat_map(|s| s.iter())
            .any(|x| !x.is_finite())
            || self.transport.iter().flatten().any(|x| !x.is_finite())
        {
            return Err("nonfinite mechanical state".into());
        }
        if self
            .bound
            .iter()
            .chain(&self.resource)
            .chain(&self.population)
            .any(|&x| !(0.0..=1.0).contains(&x))
        {
            return Err("occupancy or resource outside its mechanical bounds".into());
        }
        Ok(self)
    }

    fn density(&self) -> [f64; WIDTH] {
        let energy: f64 = self.amplitudes.iter().map(|x| x * x + 1e-12).sum();
        self.amplitudes.map(|x| (x * x + 1e-12) / energy)
    }

    fn mix(&mut self, i: usize, j: usize, m: [[f64; 2]; 2]) {
        let [a, b] = [self.amplitudes[i], self.amplitudes[j]];
        self.amplitudes[i] = m[0][0] * a + m[0][1] * b;
        self.amplitudes[j] = m[1][0] * a + m[1][1] * b;
        for k in 0..WIDTH {
            let [a, b] = [self.transport[i][k], self.transport[j][k]];
            self.transport[i][k] = m[0][0] * a + m[0][1] * b;
            self.transport[j][k] = m[1][0] * a + m[1][1] * b;
        }
    }

    fn rotate(&mut self, i: usize, j: usize, angle: f64) {
        let (s, c) = (angle * self.coupling[i][j]).sin_cos();
        self.mix(i, j, [[c, -s], [s, c]]);
    }

    fn phase(mut self) -> Self {
        for i in (1..WIDTH).step_by(2) {
            self.amplitudes[i] = -self.amplitudes[i];
            self.transport[i] = self.transport[i].map(|x| -x);
        }
        self
    }

    fn pairs(&mut self, stride: usize, angles: [f64; WIDTH]) {
        for (i, angle) in angles.into_iter().enumerate() {
            if i & stride == 0 {
                self.rotate(i, i + stride, angle);
            }
        }
    }

    fn interference(mut self) -> Self {
        for stride in [1, 2, 4, 8, 16] {
            for i in 0..WIDTH {
                if i & stride == 0 {
                    self.mix(
                        i,
                        i + stride,
                        [[FRAC_1_SQRT_2; 2], [FRAC_1_SQRT_2, -FRAC_1_SQRT_2]],
                    );
                }
            }
        }
        self
    }

    fn unitary(mut self) -> Self {
        let angles = std::array::from_fn(|i| {
            let j = i ^ 1;
            DT * self.resource[i] * self.resource[j] * (1.0 + self.bound[i] + self.bound[j])
        });
        self.pairs(1, angles);
        self
    }

    fn binding(mut self) -> Self {
        for (i, drive) in self.density().into_iter().enumerate() {
            self.bound[i] = relaxation(self.bound[i], drive, 0.5);
        }
        self
    }

    fn depression(mut self) -> Self {
        for i in 0..WIDTH {
            self.resource[i] = relaxation(self.resource[i], 0.5, self.bound[i]);
        }
        self
    }

    fn information(mut self) -> Self {
        let surprise = self.density().map(|p| -p.ln());
        self.pairs(
            1,
            std::array::from_fn(|i| DT * (surprise[i ^ 1] - surprise[i])),
        );
        self
    }

    fn oscillator(mut self) -> Self {
        for i in 0..WIDTH {
            self.velocity[i] +=
                DT * (self.bound[i] - self.bound[(i + 1) % WIDTH] - self.position[i]);
            self.position[i] += DT * self.velocity[i];
        }
        for pair in 0..WIDTH / 2 {
            let i = pair * 2 + 1;
            self.rotate(i, (i + 1) % WIDTH, DT * self.position[i]);
        }
        self
    }

    fn rotor(mut self) -> Self {
        for i in 0..WIDTH {
            self.momentum[i] = (self.momentum[i] + 5.0 * self.phase[i].sin()).rem_euclid(TAU);
            self.phase[i] = (self.phase[i] + self.momentum[i]).rem_euclid(TAU);
        }
        self.pairs(2, self.phase.map(|x| DT * x.sin()));
        self
    }

    fn population(mut self) -> Self {
        self.population = self.population.map(|x| 3.9 * x * (1.0 - x));
        self.pairs(4, self.population.map(|x| DT * (x - 0.5)));
        self
    }
}

// Exact flow of dx/dt = on*(1-x)-off*x during a constant-drive interval.
fn relaxation(value: f64, on: f64, off: f64) -> f64 {
    let rate = on + off;
    let equilibrium = on / rate;
    equilibrium + (value - equilibrium) * (-DT * rate).exp()
}

impl Primitive {
    pub fn apply(self, field: Field) -> Result<Field, String> {
        match self {
            Self::Interference => field.interference(),
            Self::Unitary => field.unitary(),
            Self::Binding => field.binding(),
            Self::Depression => field.depression(),
            Self::Information => field.information(),
            Self::Oscillator => field.oscillator(),
            Self::Rotor => field.rotor(),
            Self::Population => field.population(),
            Self::Phase => field.phase(),
        }
        .validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_numerical_fixtures() {
        assert!((relaxation(0.0, 0.25, 0.5) - 0.029829879539988613).abs() < 1e-15);
        assert!((relaxation(1.0, 0.5, 0.25) - 0.9701701204600114).abs() < 1e-15);
        let mut field = Field::new([1.0; WIDTH], 1).unwrap();
        field.phase[0] = 0.3;
        field.population[0] = 0.3;
        field = field.rotor().population();
        assert!((field.phase[0] - 1.7776010333066978).abs() < 1e-15);
        assert!((field.momentum[0] - 1.4776010333066978).abs() < 1e-15);
        assert!((field.population[0] - 0.819).abs() < 1e-15);
    }

    #[test]
    fn primitive_transport_preserves_norm_and_matches_amplitudes() {
        let signal = std::array::from_fn(|i| (i as f64 + 0.3).sin());
        let initial = Field::new(signal, 2).unwrap();
        for primitive in [
            Primitive::Interference,
            Primitive::Unitary,
            Primitive::Binding,
            Primitive::Depression,
            Primitive::Information,
            Primitive::Oscillator,
            Primitive::Rotor,
            Primitive::Population,
        ] {
            let field = primitive.apply(initial.clone()).unwrap();
            assert!(field.diagnostics()[0] < 1e-12);
            assert!((field.diagnostics()[1] - 1.0).abs() < 1e-12);
            for i in 0..WIDTH {
                let value: f64 = field.transport[i]
                    .iter()
                    .zip(initial.amplitudes)
                    .map(|(a, b)| a * b)
                    .sum();
                assert!((value - field.amplitudes[i]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn invalid_stimuli_fail_closed() {
        for signal in [[0.0; WIDTH], [f64::NAN; WIDTH], [f64::MAX; WIDTH]] {
            assert!(Field::new(signal, 1).is_err());
        }
    }

    #[test]
    fn local_phase_and_coupling_match_independent_fixture() {
        let mut signal = [0.0; WIDTH];
        signal[0] = 0.6;
        signal[1] = 0.8;
        let mut coupling = [[0.0; WIDTH]; WIDTH];
        coupling[0][1] = 0.8;
        coupling[1][0] = 0.8;
        let result = Field::coupled(signal, 1, coupling)
            .unwrap()
            .phase()
            .unitary();
        assert!((result.amplitudes[0] - 0.676869232484278).abs() < 1e-15);
        assert!((result.amplitudes[1] + 0.7361032822343239).abs() < 1e-15);
        assert!(result.diagnostics()[0] < 1e-12);
        coupling[0][1] = 0.5;
        assert!(Field::coupled(signal, 1, coupling).is_err());
        coupling[1][0] = f64::NAN;
        assert!(Field::coupled(signal, 1, coupling).is_err());
    }
}
