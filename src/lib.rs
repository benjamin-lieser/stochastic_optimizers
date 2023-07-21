mod optimizers;
mod impls;

pub use optimizers::Adam;

/// makes a type be used as a parameter in an optimizer.
/// The type should represent a owned collection of scalar variables. For example `Vec<f64>` or `[f64;10]`
pub trait Parameters {
    
    /// The scalar type of the parameters, typically f64 or f32
    type Scalar;
    
    ///
    fn zip2_mut_with<F>(&mut self, arg1: &Self, arg2: &Self, f : F)
    where
        F : Fn(&mut Self::Scalar, &Self::Scalar, &Self::Scalar);
    
    /// create a new parameters object with the same shape but filled with zeros
    fn zeros(&self) -> Self;

    /// Is implemented in terms of zip2_mut_with
    fn zip_mut_with<F>(&mut self, rhs: &Self, f : F)
    where
        F : Fn(&mut Self::Scalar, &Self::Scalar)
    {
            self.zip2_mut_with(rhs, rhs, |s, a1, _| f(s,a1))
    }
    
}

/// represents common functionality shared by all optimizers
pub trait Optimizer {

    /// The Parameter type of the optimizer
    type Para;

    /// gives a reference to the parameters
    fn parameters(&self) -> &Self::Para;

    /// gives a mutable reference to the parameters. Can be used to manually update them during optimization.
    fn parameters_mut(&mut self) -> &mut Self::Para;

    /// performes on update of the optimizer with the provided gradients
    fn step(&mut self, gradients : &Self::Para);

    /// Consumes the optimizer and returns the owned parameters. Typically used at the end of optimization.
    fn into_parameters(self) -> Self::Para;
}