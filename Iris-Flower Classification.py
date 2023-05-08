#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classfication

# In[1]:


import pandas as pd
import numpy as np
a=pd.read_csv("Iris.csv")
a


# In[2]:


a.head()


# In[3]:


b=a["Species"].unique()
b


# In[4]:


a.describe()


# In[5]:


#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(data=a,x='SepalLengthCm',hue='PetalWidthCm')


# In[6]:


import matplotlib.pyplot as plt
x=a["PetalLengthCm"].value_counts()
plt.plot(x)


# In[7]:


Features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm","Species"]
Features


# In[8]:


p=a[Features]
p.head()


# In[9]:


#visualise the whole dataset
import seaborn as sns
sns.pairplot(p)


# In[10]:


#spliting
print("Enter the splitting factor ratio between train and test")
splitFactor=float(input())


# In[11]:


Iris_Features=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]
Iris_Features


# In[12]:


x=a[Iris_Features]
x.head()


# In[13]:


y=a.Species
y.head()


# In[14]:


import math
n_train = math.floor(splitFactor * x.shape[0])
n_test = math.ceil((1-splitFactor) * x.shape[0])
X_train = x[:n_train]
y_train = y[:n_train]
X_test = x[n_train:]
y_test = y[n_train:]
print("Total Number of rows in train:",X_train.shape[0])
print("Total Number of rows in test:",X_test.shape[0])


# In[15]:


#before spliting
print("x:")
print(x)
print("y:")
print(y)


# In[16]:


#After spliting
print("X_train:")
print(X_train)
print("\ny_train:")
print(y_train)
print("\nX_test:")
print(X_test)
print("\ny_test:")
print(y_test)


# In[17]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
model = tree.DecisionTreeClassifier()


# In[18]:


tree= model.fit(X_train,y_train)


# In[19]:


print("Accuracy on training set:{:.3f}".format(tree.score (X_train,y_train)))
print("Accuracy on test set:{:.3f}".format(tree.score (X_test,y_test)))


# In[20]:


y_pred=tree.predict(X_test) 
print("Test set prediction: \n{}".format(y_pred))


# In[21]:


print("Test set score:{:.2f}".format(np.mean (y_pred==y_test)))
print("Test set score:{:.2f}".format(tree.score (X_test,y_test)))


# In[22]:


features=np.array([[4.6,3.4,1.4,0.3]])
features


# In[ ]:


`

