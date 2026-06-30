<p align="center">
  <h2 align="center"> <img src="https://github.com/ranrhuang/ranrhuang.github.io/raw/master/nas3r/static/images/icon.png" width="80" align="absmiddle"> From None to All:  <br> Self-Supervised 3D Reconstruction via Novel View Synthesis </h2>
 <p align="center">
    <a href="https://ranrhuang.github.io/">Ranran Huang</a>
    ·
    <a href="https://github.com/Roldbach/">Weixun Luo</a>
    ·
    <a href="https://yebulabula.github.io/">Ye Mao</a>
    ·
    <a href="https://www.imperial.ac.uk/people/k.mikolajczyk">Krystian Mikolajczyk</a>
  </p>
  <h3 align="center"><a href="https://arxiv.org/abs/2603.27455">Paper</a> | <a href="https://ranrhuang.github.io/nas3r/">Project Page</a>  </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://github.com/ranrhuang/ranrhuang.github.io/raw/master/nas3r/static/images/teaser.jpg" alt="Teaser" width="90%">
  </a>
</p>


<p align="center">
<strong>NAS3R</strong> is a self-supervised feed-forward framework that jointly learns explicit 3D geometry and camera parameters with no ground-truth annotations and no pretrained priors.




<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a>
    </li>
    <li>
      <a href="#camera-conventions">Camera Conventions</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#running-the-code">Running the Code</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
</ol>
</details>

## Installation

1. Clone NAS3R.
```bash
git clone --recurse-submodules git@github.com:ranrhuang/NAS3R.git
cd NAS3R
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n nas3r python=3.11 -y
conda activate nas3r
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-build-isolation
pip install -e submodules/diff-gaussian-rasterization --no-build-isolation
```


## Pre-trained Checkpoints
Our models are hosted on [Hugging Face](https://huggingface.co/RanranHuang/NAS3R) 🤗

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Arch.</th>
      <th>Init.</th>
      <th>Resolution</th>
      <th>Data</th>
      <th>Views</th>
      <th>Labels</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="7"><strong>NAS3R supports fully self-supervised training from random initialization, without any 3D labels.</strong></td></tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r.ckpt">re10k_nas3r.ckpt</a></td>
      <td>VGGT</td><td>Random</td><td>224x224</td><td>RE10K</td><td>2</td><td>&#10007;</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r_multiview.ckpt">re10k_nas3r_multiview.ckpt</a></td>
      <td>VGGT</td><td>Random</td><td>224x224</td><td>RE10K</td><td>2-10</td><td>&#10007;</td>
    </tr>
    <tr><td colspan="7"><strong>NAS3R can also use pretrained weights for initialization while remaining self-supervised with no 3D labels.</strong></td></tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r_pretrained.ckpt">re10k_nas3r_pretrained.ckpt</a></td>
      <td>VGGT</td><td>VGGT</td><td>224x224</td><td>RE10K</td><td>2</td><td>&#10007;</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r_pretrained_multiview.ckpt">re10k_nas3r_pretrained_multiview.ckpt</a></td>
      <td>VGGT</td><td>VGGT</td><td>224x224</td><td>RE10K</td><td>2-10</td><td>&#10007;</td>
    </tr>
    <tr><td colspan="7"><strong>When camera intrinsics are available, models ending in <code>-I</code> can take them as an additional input.</strong></td></tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r_pretrained-I.ckpt">re10k_nas3r_pretrained-I.ckpt</a></td>
      <td>VGGT</td><td>VGGT</td><td>224x224</td><td>RE10K</td><td>2</td><td>Intrinsics</td>
    </tr>
    <tr><td colspan="7"><strong>NAS3R is architecture-flexible and also supports a Multi-view MASt3R-style variant, denoted by <code>nas3r-m</code>.</strong></td></tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r-m.ckpt">re10k_nas3r-m.ckpt</a></td>
      <td>MASt3R</td><td>Random</td><td>256x256</td><td>RE10K</td><td>2</td><td>&#10007;</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r-m_pretrained.ckpt">re10k_nas3r-m_pretrained.ckpt</a></td>
      <td>MASt3R</td><td>MASt3R</td><td>256x256</td><td>RE10K</td><td>2</td><td>&#10007;</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/RanranHuang/NAS3R/resolve/main/re10k_nas3r-m_pretrained-I.ckpt">re10k_nas3r-m_pretrained-I.ckpt</a></td>
      <td>MASt3R</td><td>MASt3R</td><td>256x256</td><td>RE10K</td><td>2</td><td>Intrinsics</td>
    </tr>
  </tbody>
</table>


We assume the downloaded weights are located in the `checkpoints` directory.



## Datasets
Please refer to [DATASETS.md](DATASETS.md) for dataset preparation.

## Running the Code
### Training


```bash
# 2 view on NAS3R (VGGT-based architecture), for multi-view training, modify view_sampler.num_context_views
python -m src.main +experiment=nas3r/random/re10k wandb.mode=online wandb.name=nas3r_re10k


# Initialized by pretrained VGGT weights for better performance and stability.
python -m src.main +experiment=nas3r/pretrained/re10k wandb.mode=online wandb.name=nas3r_re10k_pretrained

# Initialized by pretrained VGGT weights and incorporate GT intrinsics for better performance and stability. 
python -m src.main +experiment=nas3r/pretrained/re10k-I wandb.mode=online wandb.name=nas3r_re10k_pretrained-I

```

### Evaluation

#### Novel View Synthesis and Pose Estimation on NAS3R (VGGT-based architecture)
```bash
# RealEstate10K on NAS3R
python -m src.main +experiment=nas3r/random/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./checkpoints/re10k_nas3r.ckpt 

# RealEstate10K on NAS3R, 10 view
python -m src.main +experiment=nas3r/random/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    dataset.re10k.view_sampler.num_context_views=10 \
    checkpointing.load=./checkpoints/re10k_nas3r_multiview.ckpt 


# RealEstate10K on NAS3R pretrained from VGGT
python -m src.main +experiment=nas3r/random/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./checkpoints/re10k_nas3r_pretrained.ckpt 

# RealEstate10K on NAS3R pretrained from VGGT, 10view
python -m src.main +experiment=nas3r/random/re10k mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    dataset.re10k.view_sampler.num_context_views=10 \
    checkpointing.load=./checkpoints/re10k_nas3r_pretrained_multiview.ckpt 

# RealEstate10K on NAS3R pretrained from VGGT, incorporate GT intrinsics
python -m src.main +experiment=nas3r/random/re10k-I mode=test wandb.name=re10k \
    dataset/view_sampler@dataset.re10k.view_sampler=evaluation \
    dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k.json \
    checkpointing.load=./checkpoints/re10k_nas3r_pretrained-I.ckpt 


```

## Camera Conventions
We follow the [pixelSplat](https://github.com/dcharatan/pixelsplat) camera system. The camera intrinsic matrices are normalized (the first row is divided by image width, and the second row is divided by image height).
The camera extrinsic matrices are OpenCV-style camera-to-world matrices ( +X right, +Y down, +Z camera looks into the screen).

## Acknowledgements
NAS3R evolves from and extends our previous works, [SPFSplatV2](https://github.com/ranrhuang/SPFSplatV2) and [SPFSplat](https://github.com/ranrhuang/SPFSplat). It also builds upon the excellent open-source projects [NoPoSplat](https://github.com/cvg/NoPoSplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [DUSt3R](https://github.com/naver/dust3r), and [CroCo](https://github.com/naver/croco). We sincerely thank their authors for making their work publicly available.


## Citation

```
@InProceedings{huang2026nas3r,
    author    = {Huang, Ranran and Luo, Weixun and Mao, Ye and Mikolajczyk, Krystian},
    title     = {From None to All: Self-Supervised 3D Reconstruction via Novel View Synthesis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {37358-37369}
}
```
