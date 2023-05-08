#!/usr/bin/env python
# coding: utf-8

# # Unemployment Analysis With python
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ### Importing DataSet

# In[2]:


data=pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
data.head(5)#top 5 


# In[3]:


data.tail(5) 
#last 5


# In[4]:


#ndim gives dimension of the dataset 
data.ndim


# In[5]:


#shape gives number of cols & rows of the given dataset 
data.shape


# In[6]:


#size of the dataset 
data.size


# In[7]:


#info gives information about the Dataset 
data.info()


# In[8]:


#Describe() gives mathmetical information of the columns of the only type float64 
data.describe()


# In[9]:


data.isnull() 
#returns a Detaframe object where all the values are replaced with a boolean, True for Null Values, otherwise false


# In[10]:


data.isna().sum() 
# returns the no.of missing values in the Dataset


# In[11]:


#Rename the columns names for easy to understand
data.columns=['State','Data','Frequency','EUR','EE','ELPR','Region','longitude','latitude']


# In[12]:


data


# In[13]:


#datatypes of the columns 
data.dtypes


# In[14]:


data.nunique() #returns number of unique elements in the object


# In[15]:


data.max()


# In[16]:


data.min()


# ### Data Visualization

# In[17]:


x=data['State']
x


# In[18]:


y=data["EUR"]
y


# ### Analyzing Data By Bar Graph

# In[19]:


fig=px.bar(data,x='State',y='EUR',color='State',title="Unemplyment Rate State wise)via Bar Graph")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# In[20]:


fig=px.bar(data,x='Region',y='EUR',color='State',title="Unemplyment Rate Region wise via Bar Graph")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ### Analyzing Data By Box plot

# In[21]:


fig=px.box(data,x='State',y='EUR',color='State',title="Unemplyment Rate State wise)via Box plot")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ### Analyzing Data By Scatter plot

# In[22]:


fig=px.scatter(data,x='State',y='EUR',color='State',title="Unemplyment Rate State wise)via scatter plot")
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()


# ### Analyzing Data By Histplot

# In[23]:


#Look at the estimated number of emplyees according to different regions of India 
plt.title("Indian Unemployment") 
sns.histplot(x="EE", hue="Region", data=data)
plt.show()


# In[24]:


#see the unemployment rate according to different regions of India
plt.title("Indian Unemployment Rate")
sns.histplot(x="EUR", hue="Region",data=data)
plt.show()


# In[25]:


#Create a dashboard to analyze the unemployment rate of each Indian State by region
unemployment=data[["State", "Region","EUR"]]
fig=px.sunburst(unemployment, path=["Region", "State"], values="EUR",width=500, height=500, color_continuous_scale="RdYiGn",title="Unemployment rate in india")
fig.show()


# In[ ]:




