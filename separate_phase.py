# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 18:58:54 2019

@author: y.wei
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:18:28 2019

@author: y.wei
"""

import pandas as pd
import re
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from pandas import DataFrame
import seaborn as sns
import matplotlib

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
    return ions,rrngs

def atom_filter(x, Atom_range):
    Atom_total = pd.DataFrame()
    for i in range(len(Atom_range)):
        Atom = x[x['Da'].between(Atom_range['lower'][i], Atom_range['upper'][i], inclusive=True)]
        Atom_total = Atom_total.append(Atom)
        Count_Atom= len(Atom_total['Da'])   
    return Atom_total, Count_Atom  
#%%read range file
rrange_file = 'data/67538.rrng'
ions, rrngs = read_rrng(rrange_file)
Ni_range = rrngs[rrngs['comp']=='Ni:1']
Co_range = rrngs[rrngs['comp']=='Co:1']
Cr_range = rrngs[rrngs['comp']=='Cr:1']
Ti_range = rrngs[rrngs['comp']=='Ti:1']
Nb_range = rrngs[rrngs['comp']=='Nb:1']
Hf_range = rrngs[rrngs['comp']=='Hf:1']
Zr_range = rrngs[rrngs['comp']=='Zr:1']

#%%get high and low Cr
high_total = pd.DataFrame()
low_total = pd.DataFrame()
i=0
j=0
folder='predicting'
#%%for training, we seperate amorphous and fcc phase
for filename in tqdm(os.listdir(folder)):
    x = pd.read_csv(folder+'/'+filename)
    Cr_total, Count_Cr = atom_filter(x, Cr_range)    
    N_x = len(x) 
    ratio = Count_Cr/N_x  
    if ratio >= 0.1755:
        high_total=high_total.append(x)
        x.to_csv('training/fcc/{}.csv'.format(i),index=False)
        i+=1
    else:
        low_total=low_total.append(x)
        x.to_csv('training/am/{}.csv'.format(j),index=False)
        j+=1
high_total.to_csv('training/fcc.csv',index=False)
low_total.to_csv('training/am.csv',index=False)






