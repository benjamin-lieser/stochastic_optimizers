//! This module contains Parameter implementations for some types

use crate::Parameters;

impl Parameters for f32 {
    
    type Scalar = f32;

    fn zeros(&self) -> Self {
        0.0
    }

    fn zip2_mut_with<F>(&mut self, arg1: &Self, arg2: &Self, f : F)
        where
            F : Fn(&mut Self::Scalar, &Self::Scalar, &Self::Scalar) {
        f(self, arg1, arg2)
    }
}

impl Parameters for f64 {
    
    type Scalar = f64;

    fn zeros(&self) -> Self {
        0.0
    }

    fn zip2_mut_with<F>(&mut self, arg1: &Self, arg2: &Self, f : F)
        where
            F : Fn(&mut Self::Scalar, &Self::Scalar, &Self::Scalar) {
        f(self, arg1, arg2)
    }
}

impl<T> Parameters for Vec<T>
where T : Parameters
{
    type Scalar = T::Scalar;

    fn zeros(&self) -> Self {
        self.iter().map(|x| x.zeros()).collect()
    }

    fn zip2_mut_with<F>(&mut self, arg1: &Self, arg2: &Self, f : F)
        where
            F : Fn(&mut Self::Scalar, &Self::Scalar, &Self::Scalar) {
        self.iter_mut().zip(arg1.iter()).zip(arg2.iter()).for_each(|((x, a1), a2)| x.zip2_mut_with(a1, a2, &f));
    }
}

impl<T, const N : usize> Parameters for [T;N]
where T : Parameters
{
    type Scalar = T::Scalar;

    fn zeros(&self) -> Self {
        core::array::from_fn(|i| self[i].zeros())
    }

    fn zip2_mut_with<F>(&mut self, arg1: &Self, arg2: &Self, f : F)
        where
            F : Fn(&mut Self::Scalar, &Self::Scalar, &Self::Scalar) {
        self.iter_mut().zip(arg1.iter()).zip(arg2.iter()).for_each(|((x, a1), a2)| x.zip2_mut_with(a1, a2, &f));
    }
}