# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 16:00:44 2021

@author: z.rao
"""

import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split
import os
#%%read the rrng file
def read_rrng(f):
    rf = open(f,'r').readlines()
    patterns = re.compile(r'Ion([0-9]+)=([A-Za-z0-9]+).*|Range([0-9]+)=(\d+.\d+) +(\d+.\d+) +Vol:(\d+.\d+) +([A-Za-z:0-9 ]+) +Color:([A-Z0-9]{6})')
    ions = []
    rrngs = []
    for line in rf:
        m = patterns.search(line)
        if m:
            if m.groups()[0] is not None:
                ions.append(m.groups()[:2])
            else:
                rrngs.append(m.groups()[2:])
    ions = pd.DataFrame(ions, columns=['number','name'])
    ions.set_index('number',inplace=True)
    rrngs = pd.DataFrame(rrngs, columns=['number','lower','upper','vol','comp','colour'])
    rrngs.set_index('number',inplace=True) 
    rrngs[['lower','upper','vol']] = rrngs[['lower','upper','vol']].astype(float)
    rrngs[['comp','colour']] = rrngs[['comp','colour']].astype(str)
    return ions, rrngs
#%%define the elements in the results
rrange_file = 'data/67538.rrng'
ions, rrngs = read_rrng(rrange_file)
Ni_range = rrngs[rrngs['comp']=='Ni:1']
Co_range = rrngs[rrngs['comp']=='Co:1']
Cr_range = rrngs[rrngs['comp']=='Cr:1']
Ti_range = rrngs[rrngs['comp']=='Ti:1']
Nb_range = rrngs[rrngs['comp']=='Nb:1']
Hf_range = rrngs[rrngs['comp']=='Hf:1']
Zr_range = rrngs[rrngs['comp']=='Zr:1']
#%%get the atom position from the data
def get_atom(points):
    # all Atoms
    Atom = points[points.iloc[:, 3].between(Ni_range.iloc[0,0], Ni_range.iloc[0,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[1,0], Ni_range.iloc[1,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[2,0], Ni_range.iloc[2,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[3,0], Ni_range.iloc[3,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[4,0], Ni_range.iloc[4,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[5,0], Ni_range.iloc[5,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[6,0], Ni_range.iloc[6,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[7,0], Ni_range.iloc[7,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[8,0], Ni_range.iloc[8,1]) |
                  points.iloc[:, 3].between(Ni_range.iloc[9,0], Ni_range.iloc[9,1]) |
                  points.iloc[:, 3].between(Co_range.iloc[0,0], Co_range.iloc[0,1]) |
                  points.iloc[:, 3].between(Cr_range.iloc[0,0], Cr_range.iloc[0,1]) |
                  points.iloc[:, 3].between(Cr_range.iloc[1,0], Cr_range.iloc[1,1]) |
                  points.iloc[:, 3].between(Cr_range.iloc[2,0], Cr_range.iloc[2,1]) |
                  points.iloc[:, 3].between(Ti_range.iloc[0,0], Ti_range.iloc[0,1]) |
                  points.iloc[:, 3].between(Ti_range.iloc[1,0], Ti_range.iloc[1,1]) |
                  points.iloc[:, 3].between(Ti_range.iloc[2,0], Ti_range.iloc[2,1]) |
                  points.iloc[:, 3].between(Ti_range.iloc[3,0], Ti_range.iloc[3,1]) |
                  points.iloc[:, 3].between(Nb_range.iloc[0,0], Nb_range.iloc[0,1]) |
                  points.iloc[:, 3].between(Hf_range.iloc[0,0], Hf_range.iloc[0,1]) |
                  points.iloc[:, 3].between(Hf_range.iloc[1,0], Hf_range.iloc[1,1]) |
                  points.iloc[:, 3].between(Hf_range.iloc[2,0], Hf_range.iloc[2,1]) |
                  points.iloc[:, 3].between(Hf_range.iloc[3,0], Hf_range.iloc[3,1]) |
                  points.iloc[:, 3].between(Hf_range.iloc[4,0], Hf_range.iloc[4,1]) |
                  points.iloc[:, 3].between(Hf_range.iloc[5,0], Hf_range.iloc[5,1]) |
                  points.iloc[:, 3].between(Hf_range.iloc[6,0], Hf_range.iloc[6,1]) |
                  points.iloc[:, 3].between(Zr_range.iloc[0,0], Zr_range.iloc[0,1]) |
                  points.iloc[:, 3].between(Zr_range.iloc[1,0], Zr_range.iloc[1,1]) |
                  points.iloc[:, 3].between(Zr_range.iloc[2,0], Zr_range.iloc[2,1]) |
                  points.iloc[:, 3].between(Zr_range.iloc[3,0], Zr_range.iloc[3,1]) |
                  points.iloc[:, 3].between(Zr_range.iloc[4,0], Zr_range.iloc[4,1]) |
                  points.iloc[:, 3].between(Zr_range.iloc[5,0], Zr_range.iloc[5,1]) |
                  points.iloc[:, 3].between(Zr_range.iloc[6,0], Zr_range.iloc[6,1]) ]
    return Atom
#%%voxelization the atoms
def voxelize(points):
    """
    Convert `points` to centerlized voxel with size `voxel_size` and `resolution`, then padding zero to
    `padding_to_size`. The outside part is cut, rather than scaling the points.
    Args:
    `points`: pointcloud in 3D numpy.ndarray
    `voxel_size`: the centerlized voxel size, default (24,24,24)
    `padding_to_size`: the size after zero-padding, default (32,32,32)
    `resolution`: the resolution of voxel, in meters
    Ret:
    `voxel`:32*32*32 voxel occupany grid
    `inside_box_points`:pointcloud inside voxel grid
    """
    n = 8
    padding_size=(n, n, n, 1)
    voxels = np.zeros(padding_size)
    points = points.values
    origin = (np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2]))
    # set the nearest point as (0,0,0)
    points[:, 0] -= origin[0]
    points[:, 1] -= origin[1]
    points[:, 2] -= origin[2]
    
    points = pd.DataFrame(points)
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                min_x = i*(2/n)
                max_x = (i+1)*(2/n)
                min_y = j*(2/n)
                max_y = (j+1)*(2/n)
                min_z = k*(2/n)
                max_z = (k+1)*(2/n)
                Atom= points[points.iloc[:,0].between(min_x, max_x) & points.iloc[:,1].between(min_y, max_y) & 
                             points.iloc[:,2].between(min_z, max_z)]
                voxels[i,j,k, 0]=len(Atom)
    return voxels
#%%creat the dataset for training
all_data = np.empty([45454,20,20,20,1])
i=0
folder='training/fcc'
for filename in tqdm(os.listdir(folder)):
    points = pd.read_csv(folder+'/'+filename)
    points=pd.DataFrame(points)
    points=get_atom(points)
    voxels=voxelize(points)
    all_data[i] = voxels
    i+=1
folder='training/am'
for filename in tqdm(os.listdir(folder)):
    points = pd.read_csv(folder+'/'+filename)
    points=pd.DataFrame(points)
    points=get_atom(points)
    voxels=voxelize(points)
    all_data[i] = voxels
    i+=1
    
y_1 = np.empty([25772,1])
for i in tqdm(range(25772)):
    y_1[i] = 1
y_2 = np.empty([19682,1])    
for i in tqdm(range(19682)):
    y_2[i] = 0
    
y=np.append(y_1,y_2,axis=0)


all_data=all_data[0:45450]
y=y[0:45450]

bins     = np.arange(1000,25000,1000)
bin_y =  pd.DataFrame(y[:,])
y_binned = np.digitize(bin_y.index, bins, right=True)
 
X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=0.33, random_state=42, stratify=y_binned)
# with h5py.File("Fe_Al_data.h5", "a") as f:
#     del f['y_train_16']
#     print(list(f.keys()))
#%%save the results in hdf5 file
hdf5_data = h5py.File('GeWudata.h5','a')
hdf5_data.create_dataset('X_train_20',data= X_train)
hdf5_data.create_dataset('X_test_20',data= X_test)
hdf5_data.create_dataset('y_train_20',data= y_train)
hdf5_data.create_dataset('y_test_20',data= y_test)
hdf5_data.create_dataset('x_all_20_data',data= all_data)
hdf5_data.create_dataset('y_all_20_data',data= y)
hdf5_data.close()
#%%read the results in hdf5 file
import h5py
f = h5py.File('GeWudata.h5','r')
for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
    # print(f[key].value)




