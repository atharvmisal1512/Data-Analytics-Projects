#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats


# In[2]:


df=pd.read_csv('penguinedata.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


## species -- Dependent var
# all the other are independent


# In[7]:


df['Species'].value_counts()


# In[8]:


df.columns


# In[9]:


# drop the unnecessary column
df.drop(['studyName','Sample Number','Individual ID','Date Egg','Comments'],axis=1,inplace=True)


# In[10]:


df.info()


# In[11]:


for i in ['Flipper Length (mm)','Body Mass (g)','Sex']:
    print(i,df[i].unique())
    print('*************************************')


# In[12]:


df['Body Mass (g)']=np.where(df['Body Mass (g)']=='.',np.nan,df['Body Mass (g)'])
df['Sex']=np.where(df['Sex']=='.',np.nan,df['Sex'])
df['Flipper Length (mm)']=np.where(df['Flipper Length (mm)']=='.',np.nan,df['Flipper Length (mm)'])


# In[13]:


df.isnull().sum()


# In[14]:


# fill null values in sex column


# In[15]:


df['Sex'].value_counts()


# In[16]:


df['Sex']=df['Sex'].fillna('MALE')


# In[17]:


df['Flipper Length (mm)']=df['Flipper Length (mm)'].astype('float64')
df['Body Mass (g)']=df['Body Mass (g)'].astype('float64')


# In[18]:


df.plot(kind='box',subplots=True,layout=(2,2))


# In[19]:


lst=['Flipper Length (mm)','Body Mass (g)','Culmen Length (mm)','Culmen Depth (mm)']
for i in lst:
    df[i]=df[i].fillna(df[i].mean())


# In[20]:


df.isnull().sum()


# In[21]:


lst=['Species','Island','Sex']
for i in lst:
    print(df[i].value_counts())
    print('*****************************')


# In[22]:


### EDA
sns.catplot(data=df,x='Species',kind='count',height=5,aspect=2,hue='Sex')


# In[23]:


df_is=df['Island'].value_counts()


# In[24]:


df_is.plot(kind='pie',autopct='%.2f%%')


# In[25]:


sns.catplot(data=df,x='Sex',kind='count')


# In[26]:


#Numerical
df.columns


# In[27]:


sns.relplot(data=df,x='Culmen Length (mm)',y='Culmen Depth (mm)',hue='Species')


# In[28]:


sns.pairplot(df,hue='Species')


# In[29]:


### label encoding
## using label endoder for species
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])


# In[30]:


pd.set_option('display.max_rows',None)


# In[31]:


df


# In[32]:


df=pd.get_dummies(data=df,columns=['Island','Sex'],dtype='int')


# In[33]:


df.head()


# In[34]:


df.drop(['Island_Torgersen','Sex_MALE'],axis=1,inplace=True)


# In[35]:


df.head()


# In[36]:


sns.heatmap(df.corr(),annot=True)


# In[37]:


# now divided  the data into dep and ind


# In[38]:


x=df.iloc[:,1:]
y=df['Species']


# In[39]:


#### Scaling
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_scaled=ss.fit_transform(x)


# In[40]:


df_scaled=pd.DataFrame(x_scaled)


# In[41]:


## train test split
from sklearn.model_selection import train_test_split


# In[42]:


x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=101)


# In[43]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)


# In[ ]:





# In[44]:


y_pred_tree=dtree.predict(x_test)


# In[47]:


## training accuracy
from sklearn.metrics import confusion_matrix,classification_report
y_pred_train=dtree.predict(x_train)
print(accuracy_score(y_train,y_pred_train))


# In[48]:


from sklearn.metrics import accuracy_score
y_pred_test=dtree.predict(x_test)
print('testing accuracy',accuracy_score(y_test,y_pred_test))


# In[49]:


dtree.feature_importances_


# In[50]:


df_importance=pd.DataFrame()
df_importance['Column_name']=x.columns
df_importance['Importance']=dtree.feature_importances_


# In[51]:


df_importance


# In[53]:


### Plotting a tree
from sklearn.tree import plot_tree
plt.figure(figsize=(12,10))
plot_tree(dtree,filled=True,feature_names=x.columns)
plt.show()


# In[54]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='entropy')
dtree.fit(x_train,y_train)


# In[55]:


y_pred_tree=dtree.predict(x_test)


# In[56]:


## training accuracy
from sklearn.metrics import confusion_matrix,classification_report
y_pred_train=dtree.predict(x_train)
print(accuracy_score(y_train,y_pred_train))


# In[57]:


y_pred_test=dtree.predict(x_test)
print('testing accuracy',accuracy_score(y_test,y_pred_test))


# In[60]:


print(classification_report(y_test,y_pred_tree))


# In[61]:


## bagging classifier and random forest
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


# In[62]:


bag=BaggingClassifier()
bag.fit(x_train,y_train)
y_pred=bag.predict(x_test)


# In[64]:


accuracy_score(y_train,bag.predict(x_train))


# In[65]:


accuracy_score(y_test,bag.predict(x_test))


# In[66]:


rforest=RandomForestClassifier()
rforest.fit(x_train,y_train)
y_pred=rforest.predict(x_test)


# In[67]:


accuracy_score(y_train,rforest.predict(x_train))


# In[68]:


accuracy_score(y_test,rforest.predict(x_test))


# In[ ]:


## Try to find AUC  -ROC curve


# In[ ]:


# Hyperparameter tuning


# In[82]:


param={'criterion':['gini','entropy','log_loss'],'max_depth':[5,6,7,9,10],'min_samples_split':[3,4,5,6,7],'max_features':['sqrt','log2','auto']}


# In[83]:


# Grid search cv
from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier()
cv=GridSearchCV(rf,param,cv=5,scoring='accuracy')  # 5 fold validation


# In[84]:


cv.fit(x_train,y_train)


# In[85]:


cv.best_params_


# In[86]:


y_pred_cv=cv.predict(x_test)


# In[87]:


accuracy_score(y_test,y_pred_cv)


# In[90]:


## Boosting techniques
from sklearn.ensemble import AdaBoostClassifier
Ada=AdaBoostClassifier()
Ada.fit(x_train,y_train)
y_pred=Ada.predict(x_test)


# In[97]:


accuracy_score(y_train,Ada.predict(x_train))


# In[91]:


accuracy_score(y_test,y_pred)


# In[92]:


from sklearn.ensemble import GradientBoostingClassifier
grad=GradientBoostingClassifier()
grad.fit(x_train,y_train)
y_pred=grad.predict(x_test)


# In[96]:


accuracy_score(y_train,grad.predict(x_train))


# In[93]:


accuracy_score(y_test,y_pred)


# In[ ]:




