# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:44:15 2019

@author: z.rao
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import tqdm
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cmx
import os
from mpl_toolkits import mplot3d
from scipy import stats
#import mpl_scatter_density
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.stats import norm
import  pandas as pd  
#%%read the rrng file from apt   
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
#%%read the pos file from apt
def readpos(file_name):
    f = open(file_name, 'rb')
    dt_type = np.dtype({'names':['x', 'y', 'z', 'm'], 
                  'formats':['>f4', '>f4', '>f4', '>f4']})
    pos = np.fromfile(f, dt_type, -1)
    f.close()
    return pos
#%%label the elements in apt
def label_ions(pos,rrngs):
    pos['comp'] = ''
    pos['colour'] = '#FFFFFF'
    pos['nature'] = ''
    count=0;
    for n,r in rrngs.iterrows():
        count= count+1;
        pos.loc[(pos.Da >= r.lower) & (pos.Da <= r.upper),['comp','colour', 'nature']] = [r['comp'],'#' + r['colour'],count]
    
    return pos

#%%take the noise atoms out of the results
def atom_filter(x, Atom_range):
    Atom_total = pd.DataFrame()
    for i in range(len(Atom_range)):
        Atom = x[x['Da'].between(Atom_range['lower'][i], Atom_range['upper'][i], inclusive=True)]
        Atom_total = Atom_total.append(Atom)
        Count_Atom= len(Atom_total['Da'])   
    return Atom_total, Count_Atom  

#%%read the apt file   
pos_file = 'data/R5108_67538-v02.pos'
pos = readpos(pos_file)
dpos = pd.DataFrame({'x':pos['x'],
                            'y': pos['y'],
                            'z': pos['z'],
                            'Da': pos['m']})
    
dpos.to_csv('total_wu.csv',index=False)  
del dpos

from tqdm import tqdm
x_wu=pd.read_csv('total_wu.csv')
sort_x = x_wu.sort_values(by=['z'])    
print(max(sort_x['z']))
print(min(sort_x['z']))
sublength_x= int((max(sort_x['z'])-min(sort_x['z']))/50)
start = 0
end = sublength_x
#%%divide to 50 parts
for i in tqdm(range(50)):
    temp = sort_x.iloc[start:end]
    temp = sort_x[sort_x['z'].between(start, end, inclusive=True)]
    temp.to_csv('section/{}.csv'.format(i), index=False)
    start += sublength_x
    end += sublength_x
    print(end)    
#%%divide to 2 parts
part_1=sort_x[sort_x['z']<140]
part_2=sort_x[sort_x['z']>140]
part_1.to_csv('section/part_1.csv', index=False)
part_2.to_csv('section/part_2.csv', index=False)

part_1=pd.read_csv('part_1.csv')    
part_2=pd.read_csv('part_2.csv')  
    