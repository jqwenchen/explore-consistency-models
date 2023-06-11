
# Consistency Models 

Consistency Models  ([paper](https://arxiv.org/abs/2303.01469)) 


## What are Consistency Models?

Diffusion models are amazing, because they enable you to sample high fidelity + high diversity images. Downside is, you need lots of steps, something at least 20.

Progressive Distillation (Salimans & Ho, 2022) solves this with distillating 2-steps of the diffusion model down to single step. Doing this N times boosts sampling speed by $2^N$. But is this the only way? Do we need to train diffusion model and distill it $n$ times? Yang didn't think so. Consistency model solves this by mainly trianing a model to make a consistent denosing for different timesteps (Ok I'm obviously simplifying)


## Two modes of Training Consistency Models

###  distillation mode
* Logic: This method relies on using numerical ODE solvers and a pre-trained
diffusion model to generate pairs of adjacent points on a
PF ODE trajectory. By minimizing the difference between
model outputs for these pairs, we can effectively distill a
diffusion model into a consistency model, which allows generating high-quality samples with one network evaluation.

* Process: Consistency models distill the knowledge of pre-trained
diffusion models into a single-step sampler, significantly
improving other distillation approaches in sample quality, 
while allowing zero-shot image editing applications.

###  isolation mode
* Logic: second method eliminates the need for a
pre-trained diffusion model altogether, allowing us to train
a consistency model in isolation. This approach situates
consistency models as an independent family of generative
models.
* Process: Consistency models are trained in isolation, with
no dependence on pre-trained diffusion models. This makes
them an independent new class of generative models.

## How to Use

Install the package with

```bash
pip install git+https://github.com/jqwenchen/explore-consistency-models.git
```

This repo mainly implements consistency training:

There is a self-contained MNIST  and ImageNet-9 training example on the root `main.py` and `main_imagenet9.py`.

```bash
python main.py
```
```bash
python main_imagenet9.py
```

# Reference

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.01469,
  doi = {10.48550/ARXIV.2303.01469},
  
  url = {https://arxiv.org/abs/2303.01469},
  
  author = {Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  
  keywords = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Consistency Models},
  
  publisher = {arXiv},
  
  year = {2023},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2202.00512,
  doi = {10.48550/ARXIV.2202.00512},
  
  url = {https://arxiv.org/abs/2202.00512},
  
  author = {Salimans, Tim and Ho, Jonathan},
  
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Progressive Distillation for Fast Sampling of Diffusion Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
