#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing the libraries
import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats


# In[3]:


df=pd.read_csv('insurance.csv')


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[ ]:


## No preprocessing required
## if not given good result we will go for scaling


# In[6]:


## insurance claim is dependent variable
x=df.drop('insuranceclaim',axis=1)
y=df['insuranceclaim']


# In[10]:


# Train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=101)


# In[11]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)


# In[13]:


y_pred=lr.predict(x_test)


# In[14]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[15]:


confusion_matrix(y_test,y_pred)


# In[16]:


accuracy_score(y_test,y_pred)


# In[17]:


from sklearn.metrics import precision_score,recall_score
prec=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
print('precision',prec)
print('recall',recall)


# In[18]:


from sklearn.metrics import classification_report
cls=classification_report(y_test,y_pred)
print(cls)


# In[19]:


lr.predict(x_test)


# In[22]:


# predict_proba returens probablity values of class 0 and class 1
prob=lr.predict_proba(x_test)[:,1]
np.where(prob>=0.5,1,0)


# In[26]:


threshold=[0.5,0.4,0.3,0.2,0.1]
tprs=[]
fprs=[]
for i in threshold:
    y_pred=np.where(prob>=i,1,0)
    tn,fp,fn,tp=confusion_matrix(y_test,y_pred).ravel()
    fpr=fp/(tn+fp)
    tpr=tp/(fn+tp)
    tprs.append(tpr)
    fprs.append(fpr)


# In[28]:


plt.figure()
plt.plot(fprs,tprs,'o--r')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC')
plt.show()


# In[ ]:


## as FPR increases TPR is also increasing
## for good recall model
#we have to change the threshold value
# by graph we can see 0.2 giving best values for TPR and FPR


# In[29]:


y_pred=np.where(prob>0.2,1,0)
recall_score(y_test,y_pred)

