use crate::{Parameters, Optimizer};
use num_traits::Float;

#[derive(Debug)]
pub struct SGD<P : Parameters> {
    parameters : P,
    learning_rate : P::Scalar
}

impl<Scalar, P : Parameters<Scalar = Scalar>> SGD<P>
where
    Scalar : Float
{
    /// Creates a new SGD optimizer for parameters with given learning rate.
    pub fn new(parameters : P, learning_rate : Scalar) -> SGD<P> {
        SGD { parameters, learning_rate}
    }
}

impl<Scalar, P : Parameters<Scalar = Scalar>> Optimizer for SGD<P>
where
    Scalar : Float
{
    type P = P;

    fn step(&mut self, gradients : &P) {
        self.parameters.zip_mut_with(&gradients, |p,&g| *p = *p - self.learning_rate * g);
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

    fn change_learning_rate(&mut self, learning_rate : Scalar) {
        self.learning_rate = learning_rate;
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    use tch::COptimizer;

    #[test]
    fn pytorch_compare() {
        let init = vec![3.0, 1.0, 4.0, 1.0, 5.0];

        let optimizer = SGD::new(init, 0.005);

        let optimizer_torch = COptimizer::sgd(0.005, 0.0, 0.0, 0.0,false).unwrap();

        assert!(crate::test_utils::compare_optimizers(optimizer, optimizer_torch));
    }
}