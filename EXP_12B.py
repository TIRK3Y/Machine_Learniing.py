#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


data = pd.read_csv(r'D:\University\Jain\SEM 3\ML\Lab\exp12\ENJOYSPORT.csv')
print(data)


# In[3]:


print('Target Category wise Number of Records')
data.groupby('EnjoySport').size()

data.info()


# In[4]:


input_feat = np.array(data)[:,:-1]
print("The attributes are: ")
print(input_feat)

output_feat = np.array(data)[:,-1]
print("The target is: ")
print(output_feat)


# In[5]:


def Cand_Eliminate(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_hypothesis and general_hypothesis")
    print(specific_h)
    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)
    
    for i, h in enumerate(concepts):
        if target[i] == 1:
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == 0:
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
        print("Steps of Candidate Elimination Algorithm", i + 1)
        print(specific_h)
        print(general_h)

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h

s_final, g_final = Cand_Eliminate(input_feat, output_feat)

print("Final Specific Hypothesis:", s_final, sep="\n")
print("Final General Hypothesis:", g_final, sep="\n")


# In[ ]:




