#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv(r'D:\University\Jain\SEM 3\ML\Lab\exp12\ENJOYSPORT.csv')
print(data)


# In[4]:


print('Target Category wise Number of Records')
data.groupby('EnjoySport').size()

data.info()


# In[5]:


input_feat = np.array(data)[:,:-1]
print("The attributes are: ")
input_feat

output_feat = np.array(data)[:,-1]
print("The target is: ")
output_feat


# In[6]:


def find_S(c, t):
    for i, val in enumerate(t):
        if val == 1:
            specific_hypothesis = c[i].copy()
            break
    for i, val in enumerate(c):
        if t[i] == 1:
            for x in range(len(specific_hypothesis)):
                if val[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass
    return specific_hypothesis

print("The final hypothesis is:", find_S(input_feat, output_feat))


# In[ ]:




