#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd


# In[18]:


df = pd.read_csv('survey_results_public.csv')
df.shape
#df.info
df.head()


# In[19]:


df_schema = pd.read_csv('survey_results_schema.csv')
df_schema.shape, df_schema.tail()
df_schema.head()


# In[151]:


df.columns


# In[22]:


df[['Hobbyist','Country']][0:5]


# In[48]:


df[['Hobbyist','Country']].value_counts()


# In[28]:


df.Country[0:2]


# In[42]:


df.iloc[[0,2,4],[0,1,2,3,4,5]]


# In[31]:


df.loc[[0,2,4], ['Hobbyist', 'Age','Country']]
yc = df.loc[0:10,['Age1stCode','YearsCode']]


# In[49]:


df.loc[0:4, ['Hobbyist', 'Age','Country']]


# In[51]:


df.loc[0:4, 'Hobbyist':'Country']


# In[61]:


filt = (df['Country']=='Germany') 
df[filt].head()
filt = (df['Country']=='Germany') & (df['Hobbyist']=='Yes') 
df[filt].head()


# In[64]:


df.loc[filt,['Age','Hobbyist']].head()


# In[153]:


df.loc[~filt,['Age','Hobbyist']].head()


# In[143]:


df.columns = [x.upper() for x in df.columns]
df.loc[10:12][0:4]


# In[75]:


df.loc[10,'HOBBYIST']
df.loc[10,'HOBBYIST'] = "Great"
df.loc[10,'HOBBYIST']


# In[77]:


df[10,'HOBBYIST'] = "Yes" # would not work
df.loc[10,'HOBBYIST']


# In[79]:


df.loc[10:15, 'MAINBRANCH'].apply(len)


# In[88]:


def updateAges(ages):
    ages = ages * 1.1
df.loc[10:20,'AGE'].apply(updateAges)    
df.loc[10:20,'AGE']


# In[89]:


df.loc[10:20,'AGE']
df.loc[10:20,'AGE'].apply(lambda x : x*1.1)    


# In[96]:


df.loc[10:20][1:10].apply(len)


# In[100]:


df['HOBBYIST'].map({'Yes':True, 'No':False}),df['HOBBYIST']


# In[101]:


df['HOBBYIST'].replace({'Yes':True, 'No':False}),df['HOBBYIST']


# In[104]:


df.loc[100:105].sort_values(by=['AGE'],ascending=True)


# In[111]:


df.loc[10:20].sort_values(by=['AGE','CONVERTEDCOMP'],ascending=[True,False])


# In[118]:


df['CONVERTEDCOMP'].nlargest(10)


# In[145]:


df.nlargest(10,'CONVERTEDCOMP')
df.nsmallest(10,'CONVERTEDCOMP')


# In[137]:


df.median()


# In[136]:


df.describe()


# In[159]:


df['UndergradMajor'].value_counts()


# In[169]:


country_grp = df.groupby(['Country'])
country_grp.get_group('Germany')[1:5]


# In[165]:


filt = df['Country'] == 'Germany'
df.loc[filt][0:5]


# In[174]:


country_grp = df.groupby(['Country'])
country_grp['ConvertedComp'].median()


# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


filt = (df['Country']=='United States')
yc = df.loc[filt]
yc.head(5)
yc.to_csv('modified.csv')


# In[48]:


#yc.to_excel('modified.xls') # need xlwt xlrd
yc.to_json('modified.json')

