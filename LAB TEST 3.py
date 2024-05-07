#!/usr/bin/env python
# coding: utf-8

# # AJAY TIRKEY
# # USN: 22MCAR0049

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


data = pd.read_csv(r'D:\University\Jain\SEM 3\ML\Lab\exp10\Finite Words.csv', names=['Message', 'Label'])
print("The Total instances in the Dataset: ", data.shape[0])


# In[4]:


data.head()

print('Target Category wise Number of Records')
data.groupby('Label').size()

data.info()


# In[5]:


data['class'] = data.Label.map({'pos': 1, 'neg': 0})
data.head()


# In[6]:


X = data["Message"]
y = data["class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)


# In[7]:


count_vect = CountVectorizer()
X_train_dims = count_vect.fit_transform(X_train)
X_test_dims = count_vect.transform(X_test)
df = pd.DataFrame(X_train_dims.toarray(), columns=count_vect.get_feature_names_out())


# In[8]:


mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X_train_dims, y_train)


# In[9]:


y_pred = mlp.predict(X_test_dims)


# In[10]:


print("Prediction for test set:", y_pred)


# In[11]:


test = pd.DataFrame({'Actual value': y_test, 'Predicted value': y_pred})
test


# In[12]:


matrix = confusion_matrix(y_test, y_pred)
sb.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
print(classification_report(y_test, y_pred))


# In[13]:


test_stmt = [input("Enter any statement to predict :")]
test_dims = count_vect.transform(test_stmt)
pred = mlp.predict(test_dims)
if pred == 1:
    print("Statement is Positive")
else:
    print("Statement is Negative")

