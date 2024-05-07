#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


df_diabetes=pd.read_csv(r'D:\University\Jain\SEM 3\ML\Lab\Exp7\diabetes.csv')


# In[3]:


df_diabetes.head()


# In[4]:


independent_features = df_diabetes.drop("Outcome", axis=1)
dependent_feature = df_diabetes["Outcome"]
print(independent_features)


# In[5]:


num_records, num_features = df_diabetes.shape
print("Number of records:", num_records)
print("Number of features:", num_features)


# In[6]:


feature_names = df_diabetes.columns.tolist()
print("Feature names:", feature_names)


# In[7]:


df_diabetes["Outcome"].value_counts()


# In[8]:


df_diabetes.info()


# In[9]:


numerical_description = df_diabetes.describe()
print("\nNumerical description:\n", numerical_description)


# In[10]:


df_diabetes.isnull().sum()


# In[11]:


# Identify independent and dependent features
x = df_diabetes.drop("Outcome", axis=1)
y = df_diabetes["Outcome"]

# Split dataset into train and test sets (80:20 ratio), fix random state
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[12]:


# Declare and fit the Random Forest Classifier model
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=2)
rf.fit(X_train, y_train)


# In[13]:


# Predictions on the test set
y_pred = rf.predict(X_test)

# Compare actual and predicted outputs
comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison_df)


# In[14]:


# Classification report and confusion matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[15]:


# Prediction for a new unseen record
new_record = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Input features for the new record
predicted_output = rf.predict(new_record)
if predicted_output[0] == 0:
    print("The model predicts that the person does not have diabetes.")
else:
    print("The model predicts that the person has diabetes.")


# In[ ]:




