# GEASO
#### Source code and detailed tutorials for "Network models for alignment, stitching and slice-to-volume 3D reconstruction of large-scale spatially resolved slices"

####  Yu Wang, Zaiyi Liu, Xiaoke Ma

Here,  we introduce GEASO (Graph-based Elastic Alignment for Spatial-Omics data), a network-based framework for pairwise alignment, stitching and slice-to-volume 3D reconstruction. GEASO learns consistent features of spots with graph neural network, and performs elastic registration to address both global rigid transformation and local deformation of slices by fully exploiting topological structure of graph of spots. Furthermore, GEASO adopts low-rank approximation and down-sampling strategies to accelerate algorithm, enabling application of GEASO for large-scale datasets. Experiment results demonstrate that GEASO outperforms state-of-the-art baselines in alignment, stitching and 3D reconstruction of slices across various platforms, modalities and tissues, providing a versatile tool for spatial-omics data.

![GEASO workflow](docs/framework.png)



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

Last update: 07/22/2025, version 0.0.1

Please send any questions or found bugs to Xiaoke Ma [xkma@xidian.edu.cn](mailto:xkma@xidian.edu.cn).

### Reference

- Our paper is under review.
