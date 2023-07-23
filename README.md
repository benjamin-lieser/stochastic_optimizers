# stochastic_optimizers

## Implemented Algorithms
 - Adam, see [Paper](https://arxiv.org/abs/1412.6980)
 - Stochastic Gradient Descent (still missing)
 - Adagrad (still missing)
 - RMSprop (still missing)

## Parameter Trait

Everything which implements the Parameter Trait can be optimized.

This trait essentially only defines how to do x[i] = f(x[i], y[i], z[i]) for Parameter objects x, y and z and arbitrary functions f.
This allows a large class of types to be used as Parameters to be optimized.

## Workflow

This crate does not provide automatic differentiation, but relies on the user to provide the gradients to the optimizer (By using automatic differentiation, analytical gradients, MCMC, ...).
The otpimizer owns the parameters for the scope of optimization and modiefies them.

```Rust
let initial_parameters = vec![0.0;10];

let mut optimizer = Adam::new(initial_parameters, 1e-3);

for i in 0..iterations {
    let gradient = calc_gradient(optimizer.paramerters());
    optimizer.step(&gradient);
}

let final_parameters = optimizer.into_parameters();

```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.