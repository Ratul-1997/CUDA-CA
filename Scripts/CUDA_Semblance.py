# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:10:18 2022

@author: RKS
"""


# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 17:06:58 2021

@author: JB
"""

#

import numpy as np
import scipy
import math
from numba import cuda
import time
from mayavi import mlab


''' ............................................... Result Visualization ...............................................'''

def data_Visualize3D(data_cube,colormap):
    source = mlab.pipeline.scalar_field(data_cube)
    source.spacing = [1, 1, -1]

    nx, ny, nz = data_cube.shape
    mlab.pipeline.image_plane_widget(source, plane_orientation='x_axes', 
                                     slice_index=nx//2, colormap=colormap)
    mlab.pipeline.image_plane_widget(source, plane_orientation='y_axes', 
                                     slice_index=ny//2, colormap=colormap)
    mlab.pipeline.image_plane_widget(source, plane_orientation='z_axes', 
                                     slice_index=nz//2, colormap=colormap)
    mlab.xlabel("Inline")
    mlab.ylabel("Crossline")
    mlab.zlabel("Time Samples")

    mlab.scalarbar(orientation='vertical')
    mlab.outline()

    mlab.show()
''' ............................................... CUDA Semblance Algorithm ...............................................'''

@cuda.jit('void(float64[:,:,:], int16,int16, float64[:,:])')
def semblance_cuda(mat,w2,w3, out):
    j,k = cuda.grid(2)
    if (w2<=j<=mat.shape[1]-w2 and 0<=k<=mat.shape[2]-w3) :      
         crop=mat[:,j-w2:j+w2+1,k:k+w3]
         temp_squares_of_sums=0
         temp_sum_of_squares=0
         inline,xline, nsamples = crop.shape
         ntraces=inline*xline
         for r in range(nsamples):
             squares_of_sums=0
             sum_of_squares=0
             for q in range(xline):
                 for p in range(inline):
                     squares_of_sums=squares_of_sums+crop[p,q,r]
                     sum_of_squares=sum_of_squares+crop[p,q,r]**2
                 temp_squares_of_sums=temp_squares_of_sums+(squares_of_sums**2)
                 temp_sum_of_squares=temp_sum_of_squares+sum_of_squares
         sembl=(temp_squares_of_sums+0.00000000001)/ (temp_sum_of_squares+0.00000000001)/ ntraces            
         out[j,k]=sembl




def compute_semblance(data_compute,w1,w2,w3,progress=None):
    semblance_result=np.zeros(data_compute.shape)
    inline_start=int(np.floor(w1/2))
   # print(int(np.floor(w2/2)),int(np.floor(w3)),inline_start)
    w2_gpu=int(np.floor(w2/2))
    w3_gpu=int(np.floor(w3))
  #  print(type(w3_gpu))
    progress_callback = progress if progress is not None else lambda p: None
    n=(1/(data_compute.shape[0]-2*inline_start))
    for i in range(inline_start,data_compute.shape[0]-inline_start):
        small_data=data_compute[i-inline_start:i+inline_start+1,:,:]
       # print(i)
        A_global_mem = cuda.to_device(small_data)  
        C_global_mem = cuda.device_array((small_data.shape[1],small_data.shape[2]))
        threadsperblock = (16,16)
        blockspergrid_x = int(math.ceil(small_data.shape[1] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(small_data.shape[2] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x,blockspergrid_y)
        semblance_cuda[blockspergrid, threadsperblock](A_global_mem,w2_gpu,w3_gpu, C_global_mem)
        semblance_result[i,:,:]=C_global_mem.copy_to_host()
      #  print(semblance_result[i,:,:])
        progress_callback(n*(i))
       # print(value)
    return semblance_result

''' ...............................................Importing Seismic Data ...............................................'''
data=np.load("../Data/test_data.npy")
data = np.transpose(data, (0, 2, 1))
#print("data_post",data.shape)

''' ............................................... Analysis Window Initialization and Execution  ...............................................'''

w1=3
w2=3
w3=9
inline_aug= int(np.floor(w1/2))
xline_aug=int(np.floor(w2/2))
samples_aug=int(w3)
data_aug=np.zeros([data.shape[0]+2*inline_aug,data.shape[1]+2*xline_aug,data.shape[2]+samples_aug])
data_aug[inline_aug:data_aug.shape[0]-inline_aug,xline_aug:data_aug.shape[1]-xline_aug,0:data_aug.shape[2]-samples_aug]=data
t_cudastart=time.time()
result=compute_semblance(data_aug, w1,w2, w3)
t_cudaend=time.time()
result=result[inline_aug:data_aug.shape[0]-inline_aug,xline_aug:data_aug.shape[1]-xline_aug,0:data_aug.shape[2]-samples_aug]
print("CUDA Semblance computing done, Execution Time: " , ((t_cudaend-t_cudastart)))
np.save("../Output/SemblanceVolume.npy",result)

''' ............................................... Result Visualization...............................................'''

data_Visualize3D(result,'gray')
