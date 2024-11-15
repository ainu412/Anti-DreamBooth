# PDM-Pure: Effective Purification in One Simple Python Script


> PDM-Pure is from our paper: Pixel is a Barrier: Rethinking Adversarial-Based Protection for Diffusion-Based Generation [coming soon]

## TODO

- [ ] Patch-based PDM-Pure to fit all resolutions
- [ ] Make it flexible to change to strength
- [ ] Release our paper


## Introduction

It is a **simple** but **effective** purification framework to purify protective perturbations, all you need to do is to run the single file:

```
python pdm_pure.py --dataset_name [DATASET_PATH] --img_ids [IMG_IDS] --attacks [ATTACKS] 
```

e.g. `python pdm_pure.py --dataset_name celeb20 --img_ids ariana --attacks aspl`
which means the images are in folder `celeb20/ariana_aspl` and the generated images will be in `celeb20_ariana_aspl_pdmpure`

if you want to process multiple folders at the same time
e.g. `python pdm_pure.py --dataset_name celeb20 --img_ids ariana bruce taylor --attacks aspl metacloak glaze mist`


## Installation ⚙️

For installaiton please refer to [DeepFloyd IF](https://github.com/deep-floyd/IF/tree/develop), which is the strongest opensource pixel space diffusion model (PDM) we used to purify the images.

## Results 🔍

#### Protected by AdvDM [1]
<img src="demo/advdm/original.png" width="400" height="400"/><img src="demo/advdm/purified.png" width="400" height="400"/>

#### Protected by Mist [2]
<img src="demo/mist/original.png" width="400" height="400"/><img src="demo/mist/purified.png" width="400" height="400"/>


#### Protected by Glaze [3]
<img src="demo/glaze/original.png" width="400" height="400"/><img src="demo/glaze/purified.png" width="400" height="400"/>


#### Protected by SDS [4]
<img src="demo/sds/original.png" width="400" height="400"/><img src="demo/sds/purified.png" width="400" height="400"/>







## Reference
[1] Liang, Chumeng, et al. "Adversarial Example Does Good: Preventing Painting Imitation from Diffusion Models via Adversarial Examples." International Conference on Machine Learning. PMLR, 2023.

[2] Liang, Chumeng, and Xiaoyu Wu. "Mist: Towards improved adversarial examples for diffusion models." arXiv preprint arXiv:2305.12683 (2023).


[3] Shan, Shawn, et al. "Glaze: Protecting Artists from Style Mimicry by {Text-to-Image} Models." 32nd USENIX Security Symposium (USENIX Security 23). 2023.

[4] Xue, Haotian, et al. "Toward effective protection against diffusion-based mimicry through score distillation." The Twelfth International Conference on Learning Representations. 2023.
