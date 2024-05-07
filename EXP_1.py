#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


data = fetch_20newsgroups()


# In[3]:


text_categories = data.target_names
print('Categories of Text are:')
text_categories


# In[4]:


train_data = fetch_20newsgroups(subset="train", categories=text_categories)


# In[5]:


test_data = fetch_20newsgroups(subset="test", categories=text_categories)
print('No. of Training Samples = ', len(train_data.data))
print('No. of Test Sample = ', len(test_data.data))
print('No. of Categories in Text', len(text_categories))


# In[6]:


print(train_data.data[1])


# In[7]:


nb = make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[8]:


nb.fit(train_data.data, train_data.target)


# In[9]:


y_pred = nb.predict(test_data.data)


# In[10]:


test = pd.DataFrame({'Actual value': test_data.target, 'Predicted value': y_pred})
test.head()


# In[11]:


matrix = confusion_matrix(test_data.target, y_pred)
sb.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
print(classification_report(test_data.target, y_pred))


# In[13]:


def my_predictions(my_sentence, model):
    all_categories_names = np.array(data.target_names)
    prediction = nb.predict([my_sentence])
    return all_categories_names[prediction]

my_sentence = "God"
print(my_predictions(my_sentence, nb))

my_sentence = "I am using Microsoft Windows Operating System"
print(my_predictions(my_sentence, nb))

my_sentence = "India is a parliamentary secular democratic republic"
print(my_predictions(my_sentence, nb))


# In[ ]:




