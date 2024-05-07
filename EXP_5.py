#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


df_diabetes=pd.read_csv(r'D:\University\Jain\SEM 3\ML\Lab\Exp5\diabetes.csv')


# In[5]:


df_diabetes.head()


# In[6]:


independent_features = df_diabetes.columns[df_diabetes.columns != 'Outcome']
dependent_feature = 'Outcome'


# In[7]:


print(independent_features)


# In[8]:


num_records, num_features = df_diabetes.shape
print("Number of records:", num_records)
print("Number of features:", num_features)


# In[9]:


feature_names = df_diabetes.columns.tolist()
print("Feature names:", feature_names)


# In[10]:


outcome_counts = df_diabetes['Outcome'].value_counts()
print("Outcome distribution:\n", outcome_counts)


# In[11]:


df_info = df_diabetes.info()


# In[12]:


numerical_description = df_diabetes.describe()
print("\nNumerical description:\n", numerical_description)


# In[13]:


df_diabetes.isnull().sum()


# In[14]:


X = df_diabetes[independent_features]
y = df_diabetes[dependent_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Train set: ", X_train.shape[0], X_train.shape[1])
print("Test set: ", X_test.shape[0], X_test.shape[1])


# In[17]:


lda_model = LinearDiscriminantAnalysis()


# In[19]:


lda_model.fit(X_train, y_train)


# In[20]:


y_pred = lda_model.predict(X_test)
pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})


# In[21]:


classification_rep = classification_report(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", classification_rep)
print("\nConfusion Matrix:\n", confusion_mat)


# In[22]:


new_record = [[6, 145, 72, 35, 0, 43.1, 2.288, 30]]


# In[23]:


new_record_pred = lda_model.predict(new_record)
print("Predicted diabetes outcome for the new record:", new_record_pred)


# In[24]:


logreg_model = LogisticRegression()
logreg_model.fit(X_train, y_train)


# In[25]:


y_pred_logreg = logreg_model.predict(X_test)


# In[26]:


classification_rep_logreg = classification_report(y_test, y_pred_logreg)
confusion_mat_logreg = confusion_matrix(y_test, y_pred_logreg)
print("Logistic Regression - Classification Report:\n", classification_rep_logreg)
print("\nLogistic Regression - Confusion Matrix:\n", confusion_mat_logreg)


# In[27]:


accuracy_lda = (y_test == y_pred).mean()
accuracy_logreg = (y_test == y_pred_logreg).mean()
print("\nAccuracy of Linear Discriminant Analysis model:", accuracy_lda)
print("Accuracy of Logistic Regression model:", accuracy_logreg)


# In[28]:


new_record = [[6, 145, 72, 35, 0, 43.1, 2.288, 30]]
new_record_pred_logreg = logreg_model.predict(new_record)
print("\nPredicted diabetes outcome for the new record using Logistic Regression:", new_record_pred_logreg)


# In[ ]:




