# CUDA-CA: A CUDA-based Implementation of Coherence Algorithms for Detecting Seismic Faults in Large Seismic Volumes.



Authors : Ratul Kishore Saha, Tiash Ghosh, Mamata Jenamani, Aurobinda Routray, Sanjai Kumar Singh

This repository contains the source code to perform CUDA implementation of coherence algorithm and evaluation with test data.



![title](coh_github.png)

# Contents 

* "Scipts" folder contains python programs of CUDA enabled coherence algorithms for fault detection. 
* "Data" folder contains data named as "test_data.npy" which is requied to execute the algorithms. 
* "Output" folder is requied to save the results. 
* "Results" folder contains the results of the algorithms in '.npy' format on test data. 

## Setup

## prerequisites

* Windows
* NVIDIA GPU + CUDA enbled 
* Anaconda Navigator 
* Spyder 

## Getting Started 

* Install CUDA in your machine.
Link : https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/

* Install CUDA Toolkit. 
Link : https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

## Python Packages 

The code has been tested using packages of:

* pip install numpy
* pip install scipy 
* pip install vtk
* pip install mayavi
* pip install numba 

## Running  the scripts



To execute the scripts please  follow the steps:


> * Download the codes from the repository. 
> * Open the python codes from scripts folder in Spyder. 
> * Run CUDA_semblance.py to execute CUDA enabled semblance coherence algorithm. 
> * Run CUDA_EigenStructure.py to execute CUDA enabled Eigen-Structure coherence algorithm.  
> * Run CUDA_GradientStructureTensor.py to execute CUDA enabled Gradient Structure Tensor coherence algorithm. 
> * The results will be saved in the folder "output\\" in '.npy' array format. 

# License

MIT License

Copyright (c) 2022 Ratul-1997

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


```python

```
