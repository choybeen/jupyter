#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pylab as plt
from numpy import linalg as LA
a = np.array([[1,2,3,4], [5,6,7,8]], "complex")
print(a)


# In[11]:


b = np.arange(10)
print(b)
b = np.arange(1.0,20,3)
print(b)
b = np.arange(1,20,3, dtype="float")
print(b)


# In[19]:


c = np.zeros(5)
print(c)
c = np.zeros(5,dtype="int")
print(c)
c = np.zeros((2,3))
print(c)


# In[31]:


d = np.ones((2,3))
print(d)
e = np.empty((2,3))
print(e)
e = np.empty((2,3), dtype="int")
print(e)


# In[46]:


f = np.linspace(1,10,retstep=True)
print(f)
f = np.linspace(1,10,num=10,retstep=True)
print(f)
f = np.linspace(2,3,num=6)
print(f)
f = np.linspace(2,3,num=5)
print(f)
f = np.linspace(2,3,num=5,endpoint=False)
print(f)
f = np.linspace(2,3,num=5,retstep=True)
print(f)


# In[66]:


g = np.eye(5)
print(g)
g = np.eye(5,6,dtype=int)
print(g)
g = np.eye(5,6,1)
print(g)
g = np.eye(5,6,2)
print(g)


# In[70]:


g = np.identity(3,dtype=int)
print(g)


# In[103]:


r = np.random.rand()
print(r)
r = np.random.rand(2,3)
print(r)
r = np.random.randn(2,3) #normilized
print(r)
r = np.random.randn(6) #normilized
print(r)
avg = np.average(r)
print(avg)

r = np.random.randint(2,size=8)
print(r)
r = np.random.randint(1,8,size=18)
print(r)
avg = np.average(r)
print(avg, avg.astype(int))
avg.dtype


# In[101]:


n = np.array([[1,2,3],[4,5,6]])
print(n.ndim)
print(n.shape)
n = np.zeros((3,4),dtype="int")
print(n, n.shape)
print(n.size,n.dtype,n.itemsize)


# In[111]:


a = np.arange(1,11)
print(a)
b = a[1:4]
print(b)
b = a[-4:-1]  #
print(b)
b = a[-4:]  # end of arange
print(b)
print(b[-1])


# In[122]:


c = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(c)
print(c[1][2])
print(c[-2][-1])


# In[126]:


d = np.arange(1,20)
print(d)
print(d[1:8:2])
print(d[12::2])


# In[154]:


c = np.array([np.arange(1,6),np.arange(6,11),np.arange(11,16),np.arange(16,21)])
print(c)
print(c[[0,2],[2,0]])
print(c[1:,1:])
print(c[1:,1::2])
print(c[::2,1::2])


# In[145]:


a = np.array([np.arange(1,6),np.arange(6,11),np.arange(11,16),np.arange(16,21)],dtype='f')
b = a*1.2
c = a+10
print(a,b)
d = np.array([a,b,c])
print(d)
print(d[1:2:,2::,::2])


# In[163]:


a = np.arange(1,11)
print(a)
index = np.array([1,4,5])
print(index)
print(a[index])
print(a[[0,2,5,2,5,2,5,2,5]])

c = np.array([np.arange(1,6),np.arange(6,11),np.arange(11,16),np.arange(16,21)])
print(c)
print(c>4)
print(c[c>8])


# In[169]:


a = np.array([np.arange(1,6),np.arange(6,11),np.arange(11,16),np.arange(16,21)],dtype='f')
b = a*1.2
c = b-a
print(c)
d = b/a
print(d)
d = np.divide(b,a)
print(d)
d = a*b
print(d)
d = np.multiply(a,b)
print(d)


# In[181]:


a = np.array([[1,2],[3,4],[5,6]])
b = np.array([10,20])
c = a+b
print(c)
c = a*b
print(c)
a = np.array([[1],[2],[3]])
b = np.array([10,20,30])
print(a*b)


# In[186]:


a = np.array([[1,2,3,4,5,6]])
a = np.reshape(a, (3,2))
b = np.array([4,5])
print(a, a.shape)
print(b, b.shape)
c = a*b
print(c)


# In[209]:


a = np.array([[1,2,3,4,5,6]])
b = np.resize(a,(5,2))
print(b)
c = b.flatten()
print(c)
d = b.flatten(order="F")
print(d)
d = np.reshape(d,(5,2),order="F")
print(d,b)
d = np.ravel(b)
print(d)
d = np.reshape(d,(5,2))
print(d,"\n",b)


# In[228]:


a = np.arange(10).reshape(5,2)
print(a)
b = np.transpose(a)
print(b)
c = np.resize(a,(2,3,4))
print(c)
d = np.transpose(c, axes=(1,2,0))
print(d)


# In[244]:


a = np.arange(10).reshape(5,2)
print(a)
print(a.T)
a = np.resize(np.arange(10),(3,4))
print(a)
print(a.T)


# In[265]:


a = np.arange(3)
b = np.arange(3,9)
print(a, b)
c = np.concatenate((a,b))
print(c,c.shape)


# In[270]:


a = np.arange(6).reshape(2,3)
b = np.arange(6,9).reshape(1,3)
print(a, b)
c = np.concatenate((a,b))
print(c,c.shape)


# In[272]:


a = np.arange(24).reshape(4,2,3)
b = np.arange(60,66).reshape(1,2,3)
print(a, b)
c = np.concatenate((a,b))
print(c,c.shape)


# In[296]:


a = np.array([[1,2],[3,4]])
b = np.array([[5,6]])
print(a.shape, b.shape)
c = np.concatenate((a,b),axis=0)
print(c)
c = np.vstack((a,b))
print(c)
c = np.concatenate((a,b.T),axis=1)
print(c)
c = np.hstack((a,b.T))
print(c)


# In[290]:


a = np.arange(6)
b = np.arange(6,11)
c = np.arange(11,16)

d = np.hstack((a,b))
print(d)
e = np.vstack((b,c))
print(e)


# In[305]:


a = np.arange(12)
b,c,d = np.split(a,3)
print(a)
print(b,c,d)
a = np.arange(12).reshape(3,4)
b,c,d = np.split(a,3)
print(a)
print(b,c,d)
b,c,d,e = np.hsplit(a,4)
print(b,c,d,e)


# In[317]:


a = np.arange(1,8)
b = np.arange(11,13)
print(a,b)
c = np.insert(a,3,b)
print(c)
c = np.insert(a,(3,5),b)
print(c)


# In[325]:


a = np.arange(1,13).reshape(3,4)
b = np.arange(11,14)
print(a,b)
c = np.insert(a,3,b)
print(c)
c = np.insert(a,3,b,axis=1)
print(c)
c = np.insert(a,3,50,axis=0)
print(c)
c = np.insert(a,3,50,axis=1)
print(c)


# In[338]:


a = np.arange(1,13).reshape(3,4)
b = np.arange(11,14)
print(a,b)
c = np.append(a,b.T)
print(c)
b = np.arange(11,15)
print(a,b)
c = np.append(a,b.T)
print(c)
c = np.append(a,b.reshape(1,4),axis=0)
print(c)
c = np.delete(c, 1, axis=0)
print(c)


# In[357]:


a = np.arange(1,13).reshape(3,4)
b = np.arange(1,2.12,0.1).reshape(3,4)
print(a),print(b)
print(a*b)
b = np.arange(1,2.12,0.1).reshape(4,3)
c = a.dot(b)
print(c)


# In[362]:


#matrix, only 2D
a = np.matrix("1 2; 3 4")
print(a)
b = np.matrix([[1.2,2.2],[3.2,4.2]])
print(b)
print(a+b)
print(a*b)


# In[370]:


a = np.arange(1,13).reshape(3,4)
b = np.arange(1,2.12,0.1).reshape(4,3)
print(a),print(b)
c = np.matrix(a)
d = np.matrix(b)
print(c),print(d)
e = c*d
print(e)


# In[374]:


a = np.arange(1,13).reshape(3,4)
b = np.arange(1,2.12,0.1).reshape(4,3)
c = np.matrix(a)
d = np.matrix(b)
e = c*d
print(e)
i = np.linalg.inv(e)
print(i)
r = e*i
print(e*i), print(i*e)


# In[377]:


a = np.array([[3,1],[1,2]])
b = np.array([9,8])
print(a,b)
np.linalg.solve(a,b)


# In[379]:


a = np.array([[6,2,-5],[3,3,-2],[7,5,-3]])
b = np.array([13,13,26])
print(a,b)
np.linalg.solve(a,b)


# In[380]:


a = np.array([[6,2,-5],[3,3,-2],[7,5,-3]])
b = np.array([13,13,26])
print(a,b)
np.linalg.det(a)


# In[67]:


filedata = np.genfromtxt("data.txt",delimiter=',')
filedata.astype('int32')
print(filedata)
ret = filedata[filedata>30]
print(ret)
ret = filedata>30
print(ret)


# In[72]:


a = np.random.rand(10).reshape(2,5)
print(a)
np.save("tmp.npy",a)
b = np.load("tmp.npy")
print(a-b)


# In[16]:


a = np.random.rand(10)
print(a)
b = np.var(a)
print(b)
c = np.mean(a)
print(c)
d = np.sum(a)
print(d)


# In[28]:


a = np.linspace(-np.pi, np.pi, 200)
print(a[0:3], np.deg2rad(a[0:3]))
plt.plot(a, np.sin(a))
#plt.plot(a, np.tan(a))
print(3,4,np.hypot(3,4))


# In[3]:


x = np.linspace(-np.pi*2, np.pi*2,200)
y = np.cos(x)
import matplotlib.pylab as plb

x = np.linspace(-np.pi*2, np.pi*2,200)
plb.plot(x,y)


# In[37]:


a = np.array([0,1])
b = np.array([[1,2,3,4],[5,6,7,8]])
c = np.einsum("i->", a)
print(c)
c = np.einsum("ij->", b)
print(c)
c = np.einsum("i,ij->i", a,b)
print(c)
get_ipython().run_line_magic('pinfo', 'np.einsum')


# In[51]:


a = np.arange(1,5)
b = np.arange(1,3)
print(a,"\n",b)
s = np.einsum('i->',a)
print(s)
c = np.einsum('i,j->ij', a, b)
print(c)
c = np.einsum('i,j->ij', b, a)
print(c)


# In[46]:


a = np.arange(1,13).reshape(4,3)
b = np.arange(1,7).reshape(3,2)
print(a,"\n",b)
c = np.einsum('ik,kj->ij', a, b)
print(c)


# In[60]:


a = np.arange(1,7).reshape(2,3)
print(a)
t = np.einsum('ij->ji',a)
print(t)


# In[63]:


a = np.arange(1,10).reshape(3,3)
g = LA.eig(a)
print(g)
g = LA.eigvals(a)
print(g)


# In[65]:


a = np.arange(1,10).reshape(3,3)
ai = LA.inv(a)
print(ai)
r = np.dot(a,ai)
print(r)


# In[ ]:




