# stochastic_optimizers

This crate provides implementations of common stochstic gradient optimization algorithms.
They are designed to be lightweight, flexible and easy to use.

Currently implemted:
- Adam

The crate does not provide automatic differentiation, the gradient is given by the user.

## Examples

```rust
use stochastic_optimizers::{Adam, Optimizer};
//minimise the function (x-4)^2
let start = -3.0;
let mut optimizer = Adam::new(start, 0.1);

for _ in 0..10000 {
   let current_paramter = optimizer.parameters();

   // d/dx (x-4)^2
   let gradient = 2.0 * current_paramter - 8.0;

   optimizer.step(&gradient);
}

assert_eq!(optimizer.into_parameters(), 4.0);
```
The parameters are owned by the optimizer and a reference can be optained by [`parameters()`](crate::Optimizer::parameters()).
After optimization they can be optained by [`into_parameters()`](crate::Optimizer::into_parameters()).

## What types can be optimized

All types which impement the [`Parameters`](crate::Parameters) trait can be optimized.
Implementations for the standart types `f32`, `f64`, `Vec<T : Parameters>` and `[T : Parameters ; N]` are provided.

Its realativly easy to implement it for custom types, see [`Parameters`](crate::Parameters).

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