#!/usr/bin/env python
# coding: utf-8

# # https://www.guru99.com/scipy-tutorial.html

# In[11]:


import numpy as np
import scipy as sp
from scipy.fftpack import fft, ifft
import scipy.special
x = np.array([1,2,3,4])
y = fft(x)
z = ifft(y).astype('int')
print(x, z)


# In[2]:


import numpy as np
print("I like ", np.pi)


# In[5]:


import numpy as np
from scipy import io as sio
array = np.ones((4, 4))
sio.savemat('example.mat', {'ar': array}) 
data = sio.loadmat('example.mat', struct_as_record=True)
data['ar']


# In[13]:


import scipy.special
help(sp.special)	


# In[15]:


from scipy.special import cbrt
cb = cbrt([27, 64])     #Find cubic root of 27 & 64 using cbrt() function
print(cb)      #print value of cb


# In[20]:


from scipy.special import exp10
exp = exp10([1,10,2,3])    #define exp10 function and pass value in its
print(exp)


# In[27]:


from scipy.special import comb    #find combinations of 5, 2 values using comb(N, k)
com = comb(5, 2, exact = False, repetition=True)
print(com)
com = comb(5, 2, exact = False, repetition=False) 
print(com)
#help(comb)


# In[28]:


from scipy.special import perm
#find permutation of 5, 2 using perm (N, k) function
per = perm(5, 2, exact = True)
print(per)


# In[36]:


from scipy import linalg
import numpy as np
two_d_array = np.array([ [4,5], [3,2] ])   #define square matrix
print(linalg.det( two_d_array ))               #pass values to det() function
print(linalg.inv( two_d_array ))
eg_val, eg_vect = linalg.eig(two_d_array)
#get eigenvalues
print('eign value',eg_val)
#get eigenvectors
print('eign vector',eg_vect)


# In[41]:


fre  = 5 
#Sample rate
fre_samp = 50
t = np.linspace(0, 2, 2 * fre_samp, endpoint = False )
a = np.sin(fre  * 2 * np.pi * t)
figure, axis = plt.subplots()
axis.plot(t, a)
axis.set_xlabel ('Time (s)')
axis.set_ylabel ('Signal amplitude')
plt.show()


# In[42]:


from scipy import fftpack

A = fftpack.fft(a)
frequency = fftpack.fftfreq(len(a)) * fre_samp
figure, axis = plt.subplots()

axis.stem(frequency, np.abs(A))
axis.set_xlabel('Frequency in Hz')
axis.set_ylabel('Frequency Spectrum Magnitude')
axis.set_xlim(-fre_samp / 2, fre_samp/ 2)
axis.set_ylim(-5, 110)
plt.show()


# In[44]:


plt.subplot(2,1,1)
plt.plot(t,a)
plt.subplot(2,1,2)
plt.plot(frequency, np.abs(A))


# In[45]:


from scipy import optimize

def function(a):
       return a*2 + 20 * np.sin(a)
plt.plot(a, function(a))
plt.show()
#use BFGS algorithm for optimization
optimize.fmin_bfgs(function, 0) 


# In[46]:


import numpy as np
from scipy.optimize import minimize
#define function f(x)
def f(x):   
    return .4*(1 - x[0])**2
  
optimize.minimize(f, [2, -1], method="Nelder-Mead")


# In[48]:


from scipy import misc
import matplotlib.pyplot as plt

face = misc.face()
plt.imshow(face)
plt.show()


# In[49]:


#Flip Down using scipy misc.face image  
flip_down = np.flipud(misc.face())
plt.imshow(flip_down)
plt.show()


# In[53]:


from scipy import ndimage, misc
panda_rotate = ndimage.rotate(face, 135)
plt.imshow(panda_rotate)
plt.show()


# In[56]:


from scipy import integrate
# take f(x) function as f
f = lambda x : x**2
#single integration with a = 0 & b = 1  
integration = integrate.quad(f, 0 , 1)
print(integration)


# In[59]:


from scipy import integrate
import numpy as np
from math import sqrt
f = lambda x, y : 64 *x*y
p = lambda x : 0
q = lambda y : sqrt(1 - 2*y**2)
integration = integrate.dblquad(f , 0 , 2/4,  p, q)
print(integration)


# In[63]:


import pandas as pd
data = pd.read_csv('csvtypedata.csv')
data.head()

