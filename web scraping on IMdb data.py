#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup as bs
import pandas as pd
import requests
from IPython.core.display import display,HTML
display(HTML())
url='https://m.imdb.com/chart/top/'
data=requests.get(url,headers={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'})
print(data)


# In[2]:


soup=bs(data.text)
#print(soup)


# In[4]:


content=soup.find('ul',{'class':'ipc-metadata-list'})
print(content.prettify())


# In[8]:


title_list=[]
for i in content.find_all('h3'):
    #print(i.text)
    title_list.append(i.text)
print(title_list)


# In[10]:


final_title_list=[]
for i in title_list:
    i=i.split(maxsplit=1)
    final_title_list.append(i[1])    
    
print(final_title_list)


# In[20]:


data1=[]
for i in content.find_all('div',{'class':'sc-b189961a-7'}):
    data1.append(i.text)
print(data1)


# In[22]:


final_year_list=[]
for i in data1:
    final_year_list.append(i[0:4]) 
print(final_year_list)
    


# In[24]:


temp_list=[]
for i in data1:
    temp_list.append(i[4:]) 
print(temp_list)
    


# In[48]:


data2=[]
for i in content.find_all('div',{'class':'class="ipc-rating-star ipc-rating-star--base ipc-rating-star--imdb ratingGroup--imdb-rating"'}):
    data1.append(i.text)
print(data2)


# In[44]:


import re
duration_list=[]
pattern=r'\b\d\d?[h]?\s?\d?\d?[m]?'
for i in temp_list:
    m=re.search(pattern,i)
    #print(m.group())
    duration_list.append(m.group())
print(duration_list)


# In[87]:


data2=[]
for i in content.find_all('span',{'class':'ratingGroup--imdb-rating'}):
    print(i.text)
    data2.append(i.text)
print(data2)

    
    


# In[88]:


df=pd.DataFrame({'Title':final_title_list,'Year':final_year_list,'Duration':duration_list})
df

