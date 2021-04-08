#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
from numpy.random import rand
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[38]:


N=100
t = np.linspace(0,np.pi*5, 100)
x = np.cos(t)
plt.figure()
plt.plot(x)
plt.title("cos")
plt.Text(0.5,1,"cos")


# In[39]:


y = np.cos(t*2)
plt.plot(t,x,y)


# In[40]:


plt.plot(t,x,'r-o',x,y,'b-+')
plt.legend(['sin(x)','sin(5x)'])


# In[42]:


e = rand(N)*0.5
e
x += e
colors = rand(N)
area = rand(N)*100
plt.scatter(t,x, s=area, c=colors)
plt.colorbar()


# In[49]:


e1 = rand(N)
e2 = rand(N)*2
e3 = rand(N)*10
e4 = rand(N)*100
corrmatrix = np.corrcoef([e1,e2,e3,e4])
plt.imshow(corrmatrix,cmap='GnBu')
plt.colorbar()
plt.savefig('plot.jpg')
get_ipython().run_line_magic('pwd', '')


# In[54]:


plt.subplot(2,1,1)
plt.plot(t,x)
plt.subplot(2,1,2)
plt.plot(t,y)


# In[55]:


plt.subplot(1,2,1)
plt.plot(t,x)
plt.subplot(1,2,2)
plt.plot(t,y)


# In[64]:


plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.plot(t,x)
plt.subplot(1,2,2)
plt.plot(t,y)


# In[91]:


from numpy.random import randint
data = randint(1000,size=(10,1000))
x=np.sum(data,axis=0)
plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.hist(x,color='r')
plt.subplot(1,2,2)
plt.hist(x,cumulative=True,color='g')


# In[79]:


plt.hist([x,x,x])


# In[83]:


plt.plot(np.cos(t),label='cos')
plt.plot(np.sin(t),label='sin')
plt.legend()


# In[86]:


plt.plot(t, np.cos(t),label='cos')
plt.plot(t, np.sin(t),label='sin')
plt.legend()
plt.title("con/sin")


# In[112]:


arr_1 = np.random.randint(1,7,5000)
arr_2 = np.random.randint(1,7,5000)
arr_3 = arr_1 + arr_2 #np.random.randint(1,7,5000)
plt.hist(arr_3,bins=11, density=True, stacked=True)


# In[90]:


x_1 = np.linspace(0,5,10)
y_1 = x_1**2
plt.plot(x_1,y_1, label='x/x2')
plt.plot(y_1,x_1, label='x2/x')
plt.title('Days squared chart')
plt.xlabel('days')
plt.ylabel('days squared')


# In[122]:


N1=10
t_2 = np.linspace(0,np.pi*5, N1)
x_2 = rand(N1)*0.5
x_var = rand(N1)*0.01
colors = rand(N1)
plt.bar(t_2,x_2,yerr=x_var)


# In[127]:


N1=10
t_2 = np.linspace(0,np.pi*5, N1)
x_2 = rand(N1)*0.5
x_3 = rand(N1)*0.3
x_var = rand(N1)*0.01
colors = rand(N1)
spc = np.arange(N1)
plt.bar(spc,x_2,width=0.45,yerr=x_var)
plt.bar(spc+0.5,x_3,width=0.45,yerr=x_var)


# In[139]:


N1=10
t_2 = np.linspace(0,np.pi*5, N1)
x_2 = rand(N1)*0.5
x_3 = 0.5-x_2 #rand(N1)*0.5
x_var = rand(N1)*0.01
colors = rand(N1)
spc = np.arange(N1)
plt.bar(spc,x_2,width=0.45,label='male',bottom=x_3)
plt.bar(spc,x_3,width=0.45,label='female')


# In[160]:


N1=7
t_2 = np.linspace(0,np.pi*5, N1)
x_2 = rand(N1)*0.5
x_3 = 0.5-x_2 #rand(N1)*0.5
x_var = rand(N1)*0.01
colors = rand(N1)
wedges, texts,autotexts = plt.pie(x_2,autopct='%1.0f%%')
types = ['a','b','c','c','e','f','g']
plt.legend(wedges,types,loc='right',bbox_to_anchor=(1,0,0.5,1))


# In[166]:


gd = pd.read_csv('goog.csv')
gd.head(5)


# In[170]:


gd_np = gd.to_numpy()
gd_np


# In[175]:


gd_cp = gd_np[:,4]
gd_cp[0:10]
plt.plot(gd_cp,label='price')
plt.xlabel('close price')


# In[192]:


from mpl_toolkits import mplot3d
fig9 = plt.figure(figsize=(8,5),dpi=100)
axis = fig9.add_axes([0.1,0.1,0.9,0.9],projection='3d')

N2=40
t_2 = np.linspace(0,np.pi*5, N2)
x_2 = rand(N2)*0.5
x_3 = rand(N2)*0.5
axis.scatter3D(t_2,x_2,x_3,c=x_3,cmap='Blues')


# In[211]:


def get_z(x,y):
    return np.sin(np.sqrt(x**2+y**2))

N2=50
x_2 = np.linspace(-16,16,N2)
x_3 = np.linspace(-16,16,N2)
x_4,y_4 = np.meshgrid(x_2,x_3)
z_4 = get_z(x_4,y_4)
fig10 = plt.figure(figsize=(8,5),dpi=100)
axis10 = fig10.add_axes([0.1,0.1,0.9,0.9],projection='3d')
axis10.contour3D(x_4,y_4,z_4, N2, cmap='Blues')


# In[212]:


fig10 = plt.figure(figsize=(8,5),dpi=100)
axis10 = fig10.add_axes([0.1,0.1,0.9,0.9],projection='3d')
axis10.plot_wireframe(x_4,y_4,z_4, cmap='Blues')


# In[213]:


fig10 = plt.figure(figsize=(8,5),dpi=100)
axis10 = fig10.add_axes([0.1,0.1,0.9,0.9],projection='3d')
axis10.plot_surface(x_4,y_4,z_4,rstride=1,cstride=1,cmap='Blues')

