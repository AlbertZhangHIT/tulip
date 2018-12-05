# Tulip
This repository implements the Lipschitz regularization methods discussed in 
[Improved robustness to adversarial examples using Lipschitz regularization of the loss](https://arxiv.org/abs/1810.00953).
Models are penalized by an estimate of their Lipschitz constant, taken as the maximum 2-norm of the model gradient with 
respect to the model input. The maximum is taken over mini-batches.

We also implement three types of adversarial training: 
1. Standard [FGSM](https://arxiv.org/abs/1412.6572), which we show is equivalent to penalizing by the average 1-norm of the model gradients
2. Gradient ascent, where images are perturbed in the direction of the model gradient. This is equivalent to penalizing by the average 2-norm of the model gradients
3. Perturbing images in the maximum entry-wise component of the gradient, which is equivalent to penalizing by the average inf-norm of the gradients.
4. [Projected Gradient Descent adversarial training/attack](https://arxiv.org/pdf/1706.06083.pdf).
## Details
Requires Python 3 and at least PyTorch 0.4.1.

### Training
Models are trained with `train.py`. Lipschitz regularization is enabled by passing 
the flag `--lip` and a scalar Lagrange multiplier. Similarly FGSM, gradient ascent, maximum-entry and PGD adversarial training
are respectively enabled with the flags `--J1`, `--J2`, `--Jinf` and `--PGDinf`, along with a Lagrange multiplier.

For example, in the paper the best (adversarially robust) models were trained with `--tanh --decay 5e-4 --lip 0.1 --J2 0.01` 
(includes weight decay and a final sigmoid layer). We didn't tune these Lagrange multipliers, 
and expect better results could be achieved with more tuning effort.

To incorporate this type of regularization into your own training scripts, we have included two modules, 
`adversarial_training.py` and `penalties.py`. The former contains all necessary functions to perturb training images, 
while the latter contains code for the Lipschitz penalty.

### Model summary
Summary statistics for a trained model are gathered with `summary.py`. This includes the estimate of the Lipschitz constant 
on the test data, the norm of the product of weight matrices, and best test error.

### Attacks
We include two attack scripts. The first, `nnattack.py`, calculates the L2 distance between each 
test image and the nearest training image with a different label.
The second, `attack.py`, is a wrapper into [Foolbox](https://github.com/bethgelab/foolbox),
and attacks all test images with one of many available adversarial attacks.

### Citation
If you find Tulip and/or this type of regularization useful in your scientific work, please consider citing it as
```
@article{finlay2018tulip,
  title={Improved robustness to adversarial examples using {L}ipschitz regularization of the loss},
  author={Finlay, Chris and Oberman, Adam and Abbasi, Bilal},
  journal={arXiv preprint arXiv:1810.00953},
  year={2018},
  url={http://arxiv.org/abs/1810.00953},
  archivePrefix={arXiv},
  eprint={1810.00953},
}
```
