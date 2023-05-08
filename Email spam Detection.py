#!/usr/bin/env python
# coding: utf-8

# # Email spam Detection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# In[2]:


#importing the dataset
a=pd.read_csv("spam.csv",encoding='ISO-8859-1')


# In[3]:


#first five rows & columns
a.head()


# In[4]:


#Last five rows & columns
a.tail()


# In[5]:


#Dimension of the dataset
a.ndim


# In[6]:


a.size


# In[7]:


a.shape


# In[8]:


a.describe()


# In[9]:


a.info()


# ## Data Cleaning and Pre-processing

# In[10]:


a.isnull()


# In[11]:


a.isna().sum()


# In[12]:


a.drop(columns=a[['Unnamed: 2','Unnamed: 3','Unnamed: 4']],axis=1,inplace=True)
print(a.head(5))


# In[13]:


a.columns=['spam/ham','sms']
a


# In[14]:


a.shape


# In[15]:


a.dtypes


# In[16]:


a.nunique()


# In[17]:


a.max()


# In[18]:


a.min()


# ## Data Formatting

# In[19]:


a['spam/ham'].value_counts()


# In[20]:


a['spam/ham']=a['spam/ham'].map({'spam' : 0,'ham' :1})
a


# In[21]:


x=a['sms']
x


# In[22]:


y=a['spam/ham']
y


# In[23]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)


# In[24]:


print("Training Set=(",x_train.shape,y_train.shape,")")
print("Testing Set=(",x_test.shape,y_test.shape,")")


# In[25]:


feature=TfidfVectorizer(min_df=1,stop_words="english",lowercase=True)
feature


# In[26]:


y_train=y_train.astype("int")
y_test=y_test.astype("int")


# In[27]:


xtrain=feature.fit_transform(x_train)
xtest=feature.transform(x_test)
xtrain,xtest


# In[28]:


model=LogisticRegression()
model.fit(xtrain,y_train)


# In[29]:


model.score (xtrain,y_train)


# In[30]:


model.score(xtest,y_test)


# In[31]:


#making predictions
y_pred=model.predict(xtest)
y_pred


# ## Predicting from random values

# In[32]:


b=["Is that seriously how you spell his name?"]
b=feature.transform(b)
predict=model.predict(b)
print(predict)


# In[33]:


#Evolution of model
print("Mean Squared Error: %.2f" % np.mean(y_pred-y_test)**2)


# In[ ]:




