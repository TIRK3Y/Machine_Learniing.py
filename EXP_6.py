#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report, confusion_matrix


# In[21]:


df_diabetes=pd.read_csv(r'D:\University\Jain\SEM 3\ML\Lab\Exp6\diabetes.csv')


# In[22]:


df_diabetes.head()


# In[23]:


independent_features = df_diabetes.drop("Outcome", axis=1)
dependent_feature = df_diabetes["Outcome"]
print(independent_features)


# In[24]:


num_records, num_features = df_diabetes.shape
print("Number of records:", num_records)
print("Number of features:", num_features)


# In[25]:


feature_names = df_diabetes.columns.tolist()
print("Feature names:", feature_names)


# In[26]:


feature_names = df_diabetes.columns.tolist()
print("Feature names:", feature_names)


# In[27]:


df_diabetes["Outcome"].value_counts()


# In[28]:


df_diabetes.info()


# In[29]:


numerical_description = df_diabetes.describe()
print("\nNumerical description:\n", numerical_description)


# In[30]:


df_diabetes.isnull().sum()


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(independent_features, dependent_feature, test_size=0.2, random_state=42)


# In[32]:


x = df_diabetes.drop("Outcome", axis=1)
y = df_diabetes["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Train set: ", X_train.shape[0], X_train.shape[1])
print("Test set: ", X_test.shape[0], X_test.shape[1])


# In[33]:


model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)


# In[42]:


plt.figure(figsize=(15, 10))
tree.plot_tree(model, feature_names=x.columns, class_names=["No Diabetes", "Diabetes"], filled=True)
plt.show()


# In[38]:


y_pred = model.predict(X_test)


# In[39]:


comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(comparison_df)


# In[40]:


print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[41]:


new_record = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])  # Input features for the new record
predicted_output = model.predict(new_record)
if predicted_output[0] == 0:
    print("The model predicts that the person does not have diabetes.")
else:
    print("The model predicts that the person has diabetes.")


# In[ ]:




