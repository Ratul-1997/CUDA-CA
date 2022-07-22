# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:15:37 2022

@author: JB
"""


import numpy as np
import scipy
from scipy.ndimage import gaussian_filter
import math
import time
from numba import cuda
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
''' ............................................... CUDA Gradient Structure Tensor Coherence Algorithm...............................................'''

@cuda.jit(device=True)
def matmul(A, B, C):
    """Perform square matrix multiplication of C = A * B """
    for s in range(len(A)): 
        for t in range(len(B[0])): 
            C[s,t]=0
           # print(C[s,t])
            for u in range(len(B)): 
                C[s,t] = C[s,t]+ A[s,u]*B[u,t] 
            #print(C[s,t])       
    return C
@cuda.jit(device=True)
def dot(X,Y,Y_new):  
    #print(X.shape,Y.shape,Y_new.shape)
    for v in range(X.shape[0]):
        temp1=0
        for w in range (X.shape[1]):
            temp1=temp1+X[v][w]*Y[w][0]
           # print('temp,',Y)
        Y_new[v,0]=temp1
    return Y_new    
@cuda.jit(device=True)
def norm(Z):
    temp2=0
    for x in range(Z.shape[0]):
        temp2=temp2+Z[x,0]**2
    return temp2**0.5  

@cuda.jit
def coherence_gst(mat,eig_vec_1,eig_vec_dammy_1,flat_1,cov_1,w2,w3,out):
    j,k = cuda.grid(2)
    if (w2<=j<=mat.shape[1]-w2 and 0<=k<=mat.shape[2]-w3) :      
    #for j in range(w2,mat.shape[1]-w2):
     #  for k in range(mat.shape[2]-w3):
         crop=mat[:,j-w2:j+w2+1,k:k+w3,:] 
         eig_vec=eig_vec_1[:,:,j,k]
         eig_vec_dammy=eig_vec_dammy_1[:,:,j,k]
         flat=flat_1[:,:,j,k]
         cov=cov_1[:,:,j,k]
         a=0
         for p in range(crop.shape[0]):
             for q in range(crop.shape[1]):
                 for r in range(crop.shape[2]):
                     for s in range(crop.shape[3]):  
                         flat[a,s]=crop[p,q,r,s]
                     a=a+1
     
         cov_matrix=matmul(flat.T,flat,cov) 
         
         trace=0 
         det = cov_matrix[0,0]*(cov_matrix[1,1]*cov_matrix[2,2] -cov_matrix[2,1]*cov_matrix[1,2]) - cov_matrix[0,1]*(cov_matrix[1,0]*cov_matrix[2,2] - cov_matrix[2,0]*cov_matrix[1,2]) + cov_matrix[0,2]*(cov_matrix[1,0]*cov_matrix[2,1] - cov_matrix[2,0]*cov_matrix[1,1])      
                                 
         for y in range(cov_matrix.shape[0]):
             trace=trace+cov_matrix[y,y]
         for g in range(5):
             eig_vec_update=matmul(cov_matrix,eig_vec,eig_vec_dammy)
             eig_norm=norm(eig_vec_update)
             for h in range(eig_vec.shape[0]):
                 eig_vec[h,0]=(eig_vec_update[h,0])/(eig_norm)
         
       
         eig_vec_norm=norm(eig_vec)
         matrix_dot=matmul(cov_matrix,eig_vec,eig_vec_dammy)
         matrix_dot_norm=norm(matrix_dot)
         lam1 = (matrix_dot_norm)/(eig_vec_norm)   
         vara1 = trace - lam1
         vara2 = (det)/(lam1)
         lam3 = (((vara1*vara1)-(4*vara2))**0.5 +vara1 )/2 
         lam2 = vara1 - lam3 
         eig2 = 0 
         if (lam3>=lam2):
             eig2 = lam3
         else : eig2 = lam2
         out[j,k] =  abs((lam1-eig2)/(lam1+eig2))
    
#############################################################################################   
    
def gradients(seismic, sigma):
    grads = []
    for axis in range(3):
        # Gaussian filter with order=1 is a gaussian gradient operator
        grad = scipy.ndimage.gaussian_filter1d(seismic, sigma, axis=axis, order=1)
        grads.append(grad[..., np.newaxis])
    return np.concatenate(grads, axis=3)
      
    

def gst_coherence_calc(data_compute,sigma):
        gst_result=np.zeros((data_compute.shape[0],data_compute.shape[1],data_compute.shape[2]))
        inline_start=int(np.floor(w1/2))
        w2_gpu=int(np.floor(w2/2))
        w3_gpu=int(np.floor(w3))
       # print(gst_result.shape)
        matrix_flat=np.zeros((w1*w2*w3,3,data_compute.shape[1],data_compute.shape[2])) 
        eig_vector=np.random.randn(3,1,data_compute.shape[1],data_compute.shape[2])
        eig_vector_dammy=np.random.randn(3,1,data_compute.shape[1],data_compute.shape[2])
        covariance=np.zeros((3,3,data_compute.shape[1],data_compute.shape[2]))
        C_result=cuda.device_array((data_compute.shape[1],data_compute.shape[2]))
        for i in range(inline_start,data_compute.shape[0]-inline_start):
            small_data=data_compute[i-inline_start:i+inline_start+1,:,:]
            grad = gradients(small_data,sigma)
            small_data_cuda = cuda.to_device(grad)
            threadsperblock = (16,16)
            blockspergrid_x = int(math.ceil(small_data.shape[1] / threadsperblock[0]))
            blockspergrid_y = int(math.ceil(small_data.shape[2] / threadsperblock[1]))
            blockspergrid = (blockspergrid_x,blockspergrid_y)
            coherence_gst[blockspergrid, threadsperblock](small_data_cuda,eig_vector,eig_vector_dammy,matrix_flat,covariance,w2_gpu,w3_gpu, C_result)
            gst_result[i,:,:]=C_result.copy_to_host()
        return gst_result
     

''' ...............................................Importing Seismic Data ...............................................'''

data=np.load("../Data/test_data.npy")
data=2*((data-np.min(data))/(np.max(data)-np.min(data)))-1
data=np.transpose(data,(0,2,1))

''' ............................................... Analysis Window Initialization and Execution...............................................'''

w1=3
w2=3
w3=9
sigma=0.5
inline_aug= int(np.floor(w1/2))
xline_aug=int(np.floor(w2/2))
samples_aug=int(w3)
data_aug=np.zeros([data.shape[0]+2*inline_aug,data.shape[1]+2*xline_aug,data.shape[2]+samples_aug])
data_aug[inline_aug:data_aug.shape[0]-inline_aug,xline_aug:data_aug.shape[1]-xline_aug,0:data_aug.shape[2]-samples_aug]=data
t_cudastart=time.time()
result=gst_coherence_calc(data_aug,sigma)
t_cudaend=time.time()
result=result[inline_aug:data_aug.shape[0]-inline_aug,xline_aug:data_aug.shape[1]-xline_aug,0:data_aug.shape[2]-samples_aug]
print("CUDA Gradient Structure Tensor computing done, Execution Time: " , (t_cudaend-t_cudastart))
np.save("../Output/GradientStructureTensorVolume.npy",result)


''' ...............................................Result Visualization...............................................'''
data_Visualize3D(result,'gray')
