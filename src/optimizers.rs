use crate::{Parameters, Optimizer};
use num_traits::Float;

/// Implements the Adam algorithm
#[derive(Debug)]
pub struct Adam<P, Scalar> {
    parameters : P,
    learning_rate : Scalar,
    beta1 : Scalar,
    beta2 : Scalar,
    epsilon : Scalar,
    timestep : Scalar,
    m0 : P,
    v0 : P
}

impl<Scalar, P : Parameters<Scalar = Scalar>> Adam<P, Scalar>
where
    Scalar : Float
{

    /// Creates a new Adam optimizer for parameters with given learning rate.
    /// It uses the default values beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
    pub fn new(parameters : P, learning_rate : Scalar) -> Adam<P, Scalar> {
        let m0 = parameters.zeros();
        let v0 = parameters.zeros();
        Adam { parameters, learning_rate, beta1: Scalar::from(0.9).unwrap(), beta2: Scalar::from(0.999).unwrap(), epsilon: Scalar::from(1e-8).unwrap(), timestep: Scalar::zero(), m0, v0}
    }

    
}

impl<Scalar, P : Parameters<Scalar = Scalar>> Optimizer for Adam<P, Scalar>
where
    Scalar : Float
{
    type Para = P;

    fn step(&mut self, gradients : &P) {
        self.timestep = self.timestep + Scalar::one();
        
        //m_t = beta_1 * m_t-1 + (1-beta_1) * g
        self.m0.zip_mut_with(gradients, |m, &g| *m = self.beta1 * *m + (Scalar::one() - self.beta1) * g);
        self.v0.zip_mut_with(gradients, |v, &g| *v = self.beta2 * *v + (Scalar::one() - self.beta2) * g * g);

        let bias_correction1 = Scalar::one() - self.beta1.powf(self.timestep);
        let bias_correction2_sqrt = (Scalar::one() - self.beta2.powf(self.timestep)).sqrt();

        let alpha_t = self.learning_rate / bias_correction1;

        self.parameters.zip2_mut_with(&self.m0, &self.v0, |p,&m,&v| *p = *p - alpha_t * m / (v.sqrt() / bias_correction2_sqrt + self.epsilon));
    }

    fn parameters(&self) -> &P {
        &self.parameters
    }

    fn parameters_mut(&mut self) -> &mut P {
        &mut self.parameters
    }

    fn into_parameters(self) -> P {
        self.parameters
    }
}



#[cfg(test)]
mod tests {
    use std::vec;
    use core::convert::TryFrom;

    use super::*;

    use tch::Tensor;
    use tch::Kind;
    use tch::COptimizer;

    #[test]
    fn qudratic_function() {
        let start = -3.0;
        let mut optimizer = Adam::new(start, 0.1);

        for _ in 0..10000 {
            let current_paramter = optimizer.parameters();

            // d/dx (x-4)^2
            let gradient = 2.0 * current_paramter - 8.0;
            optimizer.step(&gradient);
        }

        assert_eq!(optimizer.into_parameters(), 4.0);
    }

    #[test]
    fn pytorch_test() {
        let init = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let init_torch  = Tensor::from_slice(&init).requires_grad_(true);

        let mut optimizer = Adam::new(init, 0.005);

        let mut optimizer_torch = COptimizer::adam(0.005, 0.9, 0.999, 0.0, 1e-8, false).unwrap();
        optimizer_torch.add_parameters(&init_torch, 0).unwrap();

        for _ in 0..1000 {
            optimizer_torch.zero_grad().unwrap();

            let loss = (&init_torch * (&init_torch - 1.0)).sum(Kind::Double);
            loss.backward();

            let grad = Vec::<f64>::try_from(init_torch.grad()).unwrap();
            optimizer.step(&grad);

            optimizer_torch.step().unwrap();

        }

        println!("{:?}", init_torch);
        println!("{:?}", optimizer.into_parameters());

        assert!(false)

    }
}