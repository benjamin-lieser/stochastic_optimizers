use tch::COptimizer;
use tch::Tensor;
use tch::Kind;
use crate::Optimizer;

pub fn compare_optimizers(mut my : impl Optimizer<Para = Vec<f64>>, mut pytorch : COptimizer) -> bool {
    let init_torch  = Tensor::from_slice(my.parameters()).requires_grad_(true);

    pytorch.add_parameters(&init_torch, 0).unwrap();

    for _ in 0..1000 {
        pytorch.zero_grad().unwrap();

        let loss = (&init_torch * (&init_torch - 1.0)).sum(Kind::Double);
        loss.backward();

        let grad = Vec::<f64>::try_from(init_torch.grad()).unwrap();
        my.step(&grad);

        pytorch.step().unwrap();

    }

    let my_final = my.into_parameters();

    let pytorch_final = Vec::<f64>::try_from(init_torch).unwrap();

    my_final.iter().zip(pytorch_final.iter()).all(|(x,y)| (x-y).abs() < 1e-12)

}