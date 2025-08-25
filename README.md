# GEASO
#### Source code and detailed tutorials for "Network models for alignment, stitching and slice-to-volume 3D reconstruction of large-scale spatially resolved slices"

####  Yu Wang, Zaiyi Liu, Xiaoke Ma

Here,  we introduce GEASO (Graph-based Elastic Alignment for Spatial-Omics data), a network-based framework for pairwise alignment, stitching and slice-to-volume 3D reconstruction. GEASO learns consistent features of spots with graph neural network, and performs elastic registration to address both global rigid transformation and local deformation of slices by fully exploiting topological structure of graph of spots. Furthermore, GEASO adopts low-rank approximation and down-sampling strategies to accelerate algorithm, enabling application of GEASO for large-scale datasets. Experiment results demonstrate that GEASO outperforms state-of-the-art baselines in alignment, stitching and 3D reconstruction of slices across various platforms, modalities and tissues, providing a versatile tool for spatial-omics data.

![GEASO workflow](docs/framework.png)



## Update

**2025-08-24: We add figure code to draw figures**

**2025-08-25: The GEASO Python package is now released on Pypi!**

**2025-08-26: We upload tutorials for using GEASO to reconstruct 3D mouse brain.**



## Installation

Note: If you have an NVIDIA GPU, be sure to firstly install a version of PyTorch that supports it (We recommend Pytorch >= 2.0.1). When installing GEASO without install Pytorch previous, the CPU version of torch will be installed by default for you. [Here is the installation guide of PyTorch](https://pytorch.org/get-started/locally/).

#### 1. Start by using python virtual environment with [conda](https://anaconda.org/):

```
conda create --name GEASO python=3.9
conda activate GEASO
pip install GEASOpy
```

(Optional) To run the notebook files in tutorials, please ensure the Jupyter package is installed in your environment:

```
conda install -n geaso ipykernel
python -m ipykernel install --user --name geaso --display-name geaso-jupyter
```

#### 2. Clone from Github (We are developing Pypi package of GEASO, it will be released soon):

```
git clone https://github.com/xkmaxidian/GEASO
cd <your dir path>/GEASO
```



## Tutorial

1. The tutorial for slice alignment (with local non-rigid transformation) is accessible from : https://github.com/xkmaxidian/GEASO/blob/master/Tutorials/alignment_nonrigid.ipynb

2. The tutorial for slice stitch (with partial overlap) is accessible from : https://github.com/xkmaxidian/GEASO/blob/master/Tutorials/stitch_partial_overlap.ipynb





#### Compared slice-to-volume algorithms

Algorithms that are compared include: 

* [PASTE](https://github.com/raphael-group/paste)
* [PASTE2](https://github.com/raphael-group/paste2)
* [SLAT](https://github.com/gao-lab/SLAT)
* [Moscot](https://github.com/theislab/moscot)
* [CAST](https://github.com/wanglab-broad/CAST)
* [Spateo](https://github.com/aristoteleo/spateo-release)

### Contact:

We are continuing adding new features. Bug reports or feature requests are welcome.

Last update: 08/25/2025, version 0.1.1

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference

- Our paper is under review.
