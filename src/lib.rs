//! This crate provides implementations of common stochstic gradient optimization algorithms.
//! They are designed to be lightweight, flexible and easy to use.
//! 
//! Currently implemted:
//! - Adam
//! 
//! The crate does not provide automatic differentiation, the gradient is given by the user.
//! 
//! # Examples
//! 
//! ```
//! use stochastic_optimizers::{Adam, Optimizer};
//! //minimise the function (x-4)^2
//! let start = -3.0;
//! let mut optimizer = Adam::new(start, 0.1);
//!
//! for _ in 0..10000 {
//!    let current_paramter = optimizer.parameters();
//!
//!    // d/dx (x-4)^2
//!    let gradient = 2.0 * current_paramter - 8.0;
//! 
//!    optimizer.step(&gradient);
//! }
//!
//! assert_eq!(optimizer.into_parameters(), 4.0);
//! ```
//! The parameters are owned by the optimizer and a reference can be optained by [`parameters()`](crate::Optimizer::parameters()).
//! After optimization they can be optained by [`into_parameters()`](crate::Optimizer::into_parameters()).
//! 
//! # What types can be optimized
//! 
//! All types which impement the [`Parameters`](crate::Parameters) trait can be optimized.
//! Implementations for the standart types `f32`, `f64`, `Vec<T : Parameters>` and `[T : Parameters ; N]` are provided.
//! 
//! Its realativly easy to implement it for custom types, see [`Parameters`](crate::Parameters).

mod optimizers;
mod impls;

#[cfg(test)]
mod test_utils;

pub use optimizers::Adam;

/// Makes a type be used as a parameter in an optimizer.
/// The type should represent a owned collection of scalar variables. For example `Vec<f64>` or `[f64;10]`
pub trait Parameters {
    
    /// The scalar type of the parameters, typically [`f64`](core::primitive::f64) or [`f32`](core::primitive::f32)
    type Scalar;
    
    /// implements the follwoing elementwise operation: `self = f(self, arg1, arg2)`
    /// ## Example for `Vec<f64>`
    /// ```
    /// fn zip2_mut_with<F>(s : &mut Vec<f64>, arg1: &Vec<f64>, arg2: &Vec<f64>, f : F)
    /// where
    ///     F : Fn(&mut f64, &f64, &f64)
    /// {
    ///     s.iter_mut().zip(arg1.iter()).zip(arg2.iter()).for_each(|((s, a1), a2)| f(s,a1,a2))
    /// }
    /// ```
    fn zip2_mut_with<F>(&mut self, arg1: &Self, arg2: &Self, f : F)
    where
        F : Fn(&mut Self::Scalar, &Self::Scalar, &Self::Scalar);
    
    /// creates a new object with the same shape but filled with zeros
    /// ## Example for `Vec<f64>`
    /// ```
    /// fn zeros(s : &Vec<f64>) -> Vec<f64> {
    ///     vec![0.0;s.len()]
    /// }
    /// ```
    fn zeros(&self) -> Self;

    /// Is implemented in terms of [`zip2_mut_with`](crate::Parameters::zip2_mut_with)
    fn zip_mut_with<F>(&mut self, rhs: &Self, f : F)
    where
        F : Fn(&mut Self::Scalar, &Self::Scalar)
    {
        self.zip2_mut_with(rhs, rhs, |s, a1, _| f(s,a1))
    }
    
}

/// Represents common functionality shared by all optimizers
pub trait Optimizer {

    /// the Parameter type of the optimizer
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