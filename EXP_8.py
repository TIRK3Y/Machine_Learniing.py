#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[2]:


df_customers=pd.read_csv(r'D:\University\Jain\SEM 3\ML\Lab\Exp8\Mall_Customers.csv')


# In[3]:


df_customers.head()


# In[4]:


# Display the number of records and features
num_records, num_features = df_customers.shape
print("Number of records:", num_records)
print("Number of features:", num_features)


# In[5]:


# Display the feature names
feature_names = df_customers.columns.tolist()
print("Feature names:", feature_names)


# In[6]:


df_customers.info()


# In[9]:


df_customers.describe()


# In[11]:


df_customers.isnull().sum()


# In[12]:


#Create dummy variables for ‘Genre’
gender = pd.get_dummies(df_customers['Genre'], drop_first = True)
df_customers.head()


# In[13]:


#Concatenate the dummy variables to the original dataframe
df_customers = pd.concat([df_customers, gender], axis = 1)


# In[14]:


df_customers.head()


# In[15]:


# Removing 'Genre' and 'CustomerID' features
df_customers.drop(['Genre', 'CustomerID'], axis=1, inplace=True)


# In[16]:


df_customers.head()


# In[17]:


plt.subplots_adjust(right=2.0)
plt.subplot(1,3,1)
plt.scatter(df_customers["Age"], df_customers["Spending Score (1-100)"])
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.subplot(1,3,2)
plt.scatter(df_customers["Age"], df_customers["Annual Income (k$)"])
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.subplot(1,3,3)
plt.scatter(df_customers["Annual Income (k$)"], df_customers["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")


# In[18]:


# Finding optimal number of clusters – Elbow Method
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    km.fit(df_customers)
    wcss.append(km.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[19]:


#Fit the model
km = KMeans(n_clusters = 5)
km.fit(df_customers)
df_customers["clusters"] = km.labels_
df_customers.head()
print('No. of data objects in each cluster')
df_customers['clusters'].value_counts()
print('Centroids of the clusters assigned')
km.cluster_centers_


# In[20]:


plt.scatter(df_customers["Spending Score (1-100)"], df_customers["Annual Income (k$)"], c
= df_customers["clusters"])


# In[21]:


sse = km.inertia_
print('Sum of Squared Error (sse) = ', sse)


# In[ ]:




