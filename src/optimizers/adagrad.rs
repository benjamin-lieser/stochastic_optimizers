use crate::{Parameters, Optimizer};
use num_traits::{Float, AsPrimitive};

#[derive(Debug)]
pub struct AdaGrad<P : Parameters>{
    parameters : P,
    learning_rate : P::Scalar,
    learning_rate_decay : P::Scalar,
    state_sum : P,
    epsilon: P::Scalar,
    timestep: P::Scalar
}

impl<Scalar, P : Parameters<Scalar = Scalar>> AdaGrad<P>
where
    Scalar : Float + 'static,
    f64 : AsPrimitive<Scalar>
{
    /// Creates a new AdaGrad optimizer for parameters with given learning rate.
    /// It uses a weight decay of 0.0
    pub fn new(parameters : P, learning_rate : Scalar) -> AdaGrad<P> {
        let state_sum  = parameters.zeros();
        AdaGrad { parameters, learning_rate, learning_rate_decay: 0.0.as_(), state_sum, epsilon: 1e-10.as_(), timestep : 0.0.as_()}
    }
}

impl<Scalar, P : Parameters<Scalar = Scalar>> Optimizer for AdaGrad<P>
where
    Scalar : Float
{
    type P = P;

    fn step(&mut self, gradients : &P) {
        self.timestep = self.timestep + Scalar::one();
        
        let learning_rate = self.learning_rate / (Scalar::one() + (self.timestep - Scalar::one()) * self.learning_rate_decay);


        self.state_sum.zip_mut_with(gradients, |state, &g| *state = *state + g * g);
        
        self.parameters.zip2_mut_with(&self.state_sum, gradients, |p,&state,&g| *p = *p - learning_rate * g / (state.sqrt() + self.epsilon));
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


    #[test]
    fn qudratic_function() {
        let start = 3.5;
        let mut optimizer = AdaGrad::new(start, 0.5);

        for _ in 0..1000 {
            let current_paramter = optimizer.parameters();

            // d/dx (x-4)^2
            let gradient = 2.0 * current_paramter - 8.0;
            optimizer.step(&gradient);
        }

        assert_eq!(optimizer.into_parameters(), 4.0);
    }
}