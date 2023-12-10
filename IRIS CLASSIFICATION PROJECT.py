#!/usr/bin/env python
# coding: utf-8

# In[1]:


# First we import the all necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


# first we view the csv file
df = pd.read_csv(r"C:\Users\hplap\Downloads\archive (8)\Iris.csv")
df.head()


# In[3]:


df.isnull().sum() 


# In[4]:


df.describe()


# In[5]:


#we check the unique values
df['Species'].unique()


# In[6]:


#Then we replace the name into numeric value
df['Species'].replace({'Iris-setosa':'1','Iris-virginica':'2','Iris-vercicolor':'3'},inplace = True)


# In[7]:


df


# In[8]:


# then we divided into training and testing 80-20 ratio
x_train,_x_test,y_train,y_test=train_test_split(df[['SepalLengthCm','SepalWidthCm',
                                                    'PetalLengthCm','PetalWidthCm']],df['Species'],test_size = 0.2)


# In[9]:


len(x_train)


# In[10]:


len(y_test)


# In[11]:


# We use LogisticRegression
logistic = LogisticRegression()


# In[12]:


logistic


# In[13]:


# we will fit method
logistic.fit(x_train,y_train)


# In[14]:


logistic.predict(x_train)


# In[15]:


x_train


# In[16]:


y_test


# In[1]:


from sklearn.svm import SVC             # SVC STANDS FOR SUPPORT VECTOR CLASSIEFIER


# In[18]:


m = SVC(kernel='linear')
m


# In[19]:


#Predict rhe x_train score
logistic.score(x_train,y_train)


# In[32]:


# Create a pairplot to see
sns.pairplot(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']])
plt.show()


# In[46]:


sns.barplot(x='PetalLengthCm', y='Species', data=df,orient='h')

plt.title('Bar Plot of Sepal Length by Species')
plt.xlabel('PetalLengthCm')
plt.ylabel('Species')
plt.legend(title='Species',edgecolor='r')
plt.grid()
plt.xticks(rotation=45) 

# Show the plot
plt.show()


# In[ ]:




