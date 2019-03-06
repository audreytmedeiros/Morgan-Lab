# coding: utf-8
import cv2
syn = cv2.imread('Syn82.tif')
import matplotlib.pyplot as plt
plt.imshow(syn)
plt.show()
import pandas as pd
df = pd.read_table('Syn82testaz.txt')
df.head()
img_conv = 0.27 #pix/nm
df_pixels = df * img_conv
df_pixels
df_pixels = np.round(df * img_conv)
import numpy as np
df_pixels = np.round(df * img_conv)
df_pixels.head()
df_pixels
plt.imshow(syn)
plt.plot(df.iloc[:,0], df.iloc[:,1])
plt.show()
plt.imshow(syn)
plt.plot(df_pixels.iloc[:,0], df_pixels.iloc[:,1])
plt.show()
df.head()
df = pd.read_table('Syn82testaz.txt', header=None)
df.head()
df = pd.read_table('Syn82testaz.txt', header=None) * img_conv
df
df = np.round(pd.read_table('Syn82testaz.txt', header=None) * img_conv)
df.head()
df
df - df.iloc[1:,:]
df
df.iloc[1:,:]
df.iloc[1:,:]
df
df.iloc[:-1,:] - df.iloc[1:,:]
df.iloc[:-1,0] - df.iloc[1:,0]
df.head()
df.diff()
df_diff = df.diff()
df_diff
df_diff.iloc[:,0]**2 +df_diff.iloc[:,1]**2 
np.sqrt(df_diff.iloc[:,0]**2 +df_diff.iloc[:,1]**2) 
df_ed = np.sqrt(df_diff.iloc[:,0]**2 +df_diff.iloc[:,1]**2) 
df_diff
new_x, new_y = [], []
for row in df:
    print(row)
    
for row in df.iloc[:,:]:
    print(row)
    
for index, row in df.iterrows():
    print(i, row)
    
for i, row in df.iterrows():
    print(i, row)
    
for i, row in df.iterrows():
    print(i, row)
    break
    
for i, (x,y) in df.iterrows():
    print(i, x,y)
    break
    
    
for i, (x,y,z) in df.iterrows():
    print(i, x,y)
    break
    
    
for i, x in df.iterrows():
    print(i, x,y)
    break
    
    
for i in range(len(df)):
    if i == 0:
        continue
    
for i in range(len(df)):
    if i == 0:
        continue
        
    
x_fin, y_fin = [], []
for i in range(len(df)):
    x_fin.append(df.iloc[i,0])
    f_fin.append(df.iloc[i,1])
    if i == 0:
        continue
        
    
for i in range(len(df)):
    x_fin.append(df.iloc[i,0])
    y_fin.append(df.iloc[i,1])
    if i == 0:
        continue
        
    
x_fin, y_fin = [], []
for i in range(len(df)):
    x_fin.append(df.iloc[i,0])
    y_fin.append(df.iloc[i,1])
    if i == 0:
        continue
        
    
x_fin
df_ed.head()
df_diff.head()
x_fin, y_fin = [], []
for i in range(len(df)):
    x, y = df.iloc[i,:]
    x_fin.append(x)
    y_fin.append(y)
    if i == len(df):
        continue
    if df_ed.iloc[i+1] < 1.1:
        continue
    x_fin += np.linspace(x, df.iloc[i+1,0], int(df_ed[i+1])).as_list()
    y_fin += np.linspace(y, df.iloc[i+1,1], int(df_ed[i+1])).as_list()
    
x_fin, y_fin = [], []
for i in range(len(df)):
    x, y = df.iloc[i,:]
    x_fin.append(x)
    y_fin.append(y)
    if i == len(df):
        continue
    if df_ed.iloc[i+1] < 1.1:
        continue
    x_fin += list(np.linspace(x, df.iloc[i+1,0], int(df_ed[i+1])))
    y_fin += list(np.linspace(y, df.iloc[i+1,1], int(df_ed[i+1])))
    
x_fin, y_fin = [], []
for i in range(len(df)):
    x, y = df.iloc[i,:]
    x_fin.append(x)
    y_fin.append(y)
    if i == len(df) - 1:
        continue
    if df_ed.iloc[i+1] < 1.1:
        continue
    x_fin += list(np.linspace(x, df.iloc[i+1,0], int(df_ed[i+1])))
    y_fin += list(np.linspace(y, df.iloc[i+1,1], int(df_ed[i+1])))
    
x_fin
len(x_fin)
y_fin
from pathlib import Path
l
vesicle_path = Path('./vesicles')
vesicle_path.exists()
names = []
for file in vesicle_path.iterdir():
    names.append(file.name)
    
names
file.root
file.stem
names = []
for file in vesicle_path.iterdir():
    names.append(file.name)
    
names = []
for file in vesicle_path.iterdir():
    names.append(file.stem)
    
names
names_split = [name.split('-') for name in names]
names_split
v_x, v_y = [], []
for y_str, x_str in names_split:
    v_x.append(int(x_str))
    v_y.append(int(y_str))
    
v_x
v_y
V = np.array([v_x, v_y])
V
V.shape
V = np.array([v_x, v_y]).T
V
v_x
V
L
L = np.array([x_fin,y_fin]).T
L
L.shape
L
V
L
get_ipython().run_line_magic('pinfo', 'np.savetxt')
np.savetxt('V.txt', V)
l
np.savetxt('L.txt', L)
get_ipython().run_line_magic('save', '')
get_ipython().run_line_magic('history', '-o')
get_ipython().run_line_magic('save', '-o -f audrey_history.txt')
get_ipython().run_line_magic('save', '-f audrey_history.txt')
get_ipython().run_line_magic('save', "-f 'audrey_history.txt'")
