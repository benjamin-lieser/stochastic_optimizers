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

#[cfg(feature = "ndarray")]
mod ndarray {
    use ndarray::{ArrayBase, DataOwned, Dimension, Zip, DataMut};
    use num_traits::Float;

    use crate::Parameters;

    impl<Scalar, S , D> Parameters for ArrayBase<S, D>
    where
        Scalar : Float,
        S : DataOwned<Elem = Scalar> + DataMut<Elem = Scalar>,
        D : Dimension
    {
        type Scalar = Scalar;

        fn zeros(&self) -> Self {
            ArrayBase::<S, D>::zeros(self.raw_dim())
        }

        fn zip2_mut_with<F>(&mut self, arg1: &Self, arg2: &Self, f : F)
            where
                F : Fn(&mut Self::Scalar, &Self::Scalar, &Self::Scalar) {
            Zip::from(self).and(arg1).and(arg2).for_each(f);
        }

        fn zip_mut_with<F>(&mut self, rhs: &Self, f : F)
            where
                F : Fn(&mut Self::Scalar, &Self::Scalar) {
            self.zip_mut_with(rhs, f);
        }
    }

    #[cfg(test)]
    mod test {

        use ndarray::prelude::*;

        use crate::Optimizer;

        #[test]
        fn ndarray_test() {
            let init = Array1::from_vec([1.0, 2.0, 3.0].to_vec());

            let mut optimizer = crate::Adam::new(init, 0.1);

            for _ in 0..1000 {
                let gradients = optimizer.parameters() * 2.0 - 8.0;
                optimizer.step(&gradients);
            }

            assert_eq!(optimizer.into_parameters(), Array1::from_elem([3], 4.0));
        }
    }
}