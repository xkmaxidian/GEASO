# GEASO
#### Source code and detailed tutorials for "Network model for alignment, stitching and slice-to-volume 3D reconstruction of large-scale spatially resolved slices"

####  Yu Wang, Zaiyi Liu, Xiaoke Ma

Here,  we introduce GEASO (Graph-based Elastic Alignment for Spatial-Omics data), a network-based framework for pairwise alignment, stitching and slice-to-volume 3D reconstruction. GEASO learns consistent features of spots with graph neural network, and performs elastic registration to address both global rigid transformation and local deformation of slices by fully exploiting topological structure of graph of spots. Furthermore, GEASO adopts low-rank approximation and down-sampling strategies to accelerate algorithm, enabling application of GEASO for large-scale datasets. Experiment results demonstrate that GEASO outperforms state-of-the-art baselines in alignment, stitching and 3D reconstruction of slices across various platforms, modalities and tissues, providing a versatile tool for spatial-omics data.

![GEASO workflow](docs/framework.png)



## Update

**2025-08-16: We are developing python package of GEASO**

**2025-08-28: We upload tutorials for using GEASO to reconstruct 3D human lymph node slices sequenced by OpenST.**

**2025-09-02: We complete the runtime and memory usage comparison.**

**2025-12-20: We upload tutorials using GEASO to stich DLPFC slices**

**2026-04-29: We upload interpolate related code and tutorials**



## Installation

Note: If you have an NVIDIA GPU, be sure to firstly install a version of PyTorch that supports it (We recommend Pytorch >= 2.0.1). When installing GEASO without install Pytorch previous, the CPU version of torch will be installed by default for you. [Here is the installation guide of PyTorch](https://pytorch.org/get-started/locally/).

#### 1. Start by using python virtual environment with [conda](https://anaconda.org/):

```
conda create --name GEASO python=3.9
conda activate GEASO
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

3. The tutorial for stitching DLPFC slices (with partial overlap) is accessible from: https://github.com/xkmaxidian/GEASO/blob/master/Tutorials/DLPFC_Stitch.ipynb

4. The tutorial for slice-to-volume 3D reconstruction is accessible from: https://github.com/xkmaxidian/GEASO/blob/master/Tutorials/OpenST_3D_Rec.ipynb

5. The tutorial for down-sample and interpolate is accessible from: https://github.com/xkmaxidian/GEASO/blob/master/Tutorials/coarse_to_fine_alignment_usage.ipynb

   

#### Compared slice-to-volume algorithms

Algorithms that are compared include: 

* [PASTE](https://github.com/raphael-group/paste)
* [PASTE2](https://github.com/raphael-group/paste2)
* [SLAT](https://github.com/gao-lab/SLAT)
* [Moscot](https://github.com/theislab/moscot)
* [CAST](https://github.com/wanglab-broad/CAST)
* [Spateo](https://github.com/aristoteleo/spateo-release)
* [SPACEL](https://github.com/QuKunLab/SPACEL)
* [STAlign](https://github.com/JEFworks-Lab/STalign)

### Contact:

We are continuing adding new features. Bug reports or feature requests are welcome.

Last update: 04/29/2026, version 0.3.0

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference

- Please consider citing the following reference:

  - [https://www.nature.com/articles/s41467-026-71042-6](https://doi.org/10.1038/s41467-026-71042-6)
