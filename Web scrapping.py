#!/usr/bin/env python
# coding: utf-8

# In[1]:


test_html='''<html>
<head>
<title>
    My first web page
</title>

</head>
<body>
    <h1>
         This is something new
        <p>Please see following list</p>
    <ul>
    <li>python</li>
    <li>Java</li>
    <li>Ruby</li>
    <li>c++</li>
    <li>c</li>
    </ul>

    <p>These are Analytical tool</p>
    <ul class='sample'>
        <li>Advanced excel</li>
        <li>Power bi</li>
        <li>Tablaeu</li>
    </ul>

    </h1>
    <h1>Please see the below table </h1>

<table>
    <tr><td>Student Name</td> <td>Marks</td><td>Batch</td></tr>
    <tr><td>Atharv Misal</td> <td>99</td><td>308</td></tr>
    <tr><td>Shreya Dhole</td> <td>100</td><td>308</td></tr>
    <tr><td>Parth Dhole</td> <td>43</td><td>308</td></tr>
    <tr><td>Sukhada Misal</td> <td>-16</td><td>308</td></tr>
</table>





</body>









</html>'''


# In[2]:


from IPython.core.display import display,HTML
display(HTML())


# In[3]:


# request
#BeautifulSoup
# Pandas


# In[4]:


#pip install pandas


# In[5]:


#pip install bs4


# In[6]:


from bs4 import BeautifulSoup as bs
import pandas as pd


# In[7]:


soup=bs(test_html)


# In[8]:


print(soup)


# In[9]:


print(soup.text)


# In[10]:


#how to find the tags
print(soup.find('h1'))


# In[11]:


print(soup.find_all('h1'))


# In[12]:


data=soup.find_all('h1')
for i in data:
    print(i.text)


# In[13]:


print(soup.find_all('ul',{'class':'sample'}))


# In[14]:


#libraries:
import pandas as pd
import requests
from bs4 import BeautifulSoup as bs


# In[15]:


#### Web Scrapping
#requests
url='https://en.wikipedia.org/wiki/List_of_countries_by_coffee_production'
data=requests.get(url)


# In[16]:


print(data)


# In[17]:


soup=bs(data.text)


# In[18]:


print(soup)


# In[19]:


table=soup.find('table')
print(table.text)


# In[20]:


main_data=[]
for row in table.find_all('tr')[1:]:
    #print(row.text.replace('\n',' '))
    tr_data=row.text.replace('\n',' ')
    #print(tr_data)
    temp_data=tr_data.split()
    #print(temp_data)
    main_data.append(temp_data)
print(main_data)


# In[21]:


df=pd.DataFrame(main_data)
print(df)


# In[22]:


main_test=[['1', 'Brazil', '44,200,000', '2,652,000', '5,714,381,000'], ['2', 'Vietnam', '27,500,000', '1,650,000', '3,637,627,000'], ['3', 'Colombia', '13,500,000', '810,000', '1,785,744,000'], ['4', 'Indonesia', '11,000,000', '660,000', '1,455,050,000'], ['5', 'Honduras', '9,600,000', '580,000', '1,278,681,000'], ['6', 'Ethiopia', '6,400,000', '384,000', '846,575,000'], ['7', 'India', '5,800,000', '348,000', '767,208,000'], ['8', 'Uganda', '4,800,000', '288,000', '634,931,000'], ['9', 'Mexico', '3,900,000', '234,000', '515,881,000'], ['10', 'Guatemala', '3,400,000', '204,000', '449,743,000'], ['11', 'Peru', '3,200,000', '192,000', '423,287,000'], ['12', 'Nicaragua', '2,200,000', '132,000', '291,010,000'], ['13', 'China(2013–14 est.)[7]', '1,947,000', '116,820', '257,544,000'], ['14', 'Ivory Coast', '1,800,000', '108,000', '238,099,000'], ['15', 'Costa Rica', '1,492,000', '89,520', '197,357,000'], ['16', 'Kenya', '833,000', '49,980', '110,187,000'], ['17', 'Papua New Guinea', '800,000', '48,000', '105,821,000'], ['18', 'Tanzania', '800,000', '48,000', '105,821,000'], ['19', 'El Salvador', '762,000', '45,720', '100,795,000'], ['20', 'Ecuador', '700,000', '42,000', '92,594,000'], ['21', 'Cameroon', '570,000', '34,200', '75,398,000'], ['22', 'Laos', '520,000', '31,200', '68,784,000'], ['23', 'Madagascar', '520,000', '31,200', '68,784,000'], ['24', 'Gabon', '500,000', '30,000', '66,138,000'], ['25', 'Thailand', '500,000', '30,000', '66,138,000'], ['26', 'Venezuela', '500,000', '30,000', '66,138,000'], ['27', 'Dominican Republic', '400,000', '24,000', '52,910,000'], ['28', 'Haiti', '350,000', '21,000', '46,297,000'], ['29', 'Democratic Republic of the Congo', '335,000', '20,100', '44,312,000'], ['30', 'Rwanda', '250,000', '15,000', '33,069,000'], ['31', 'Burundi', '200,000', '12,000', '26,455,000'], ['32', 'Philippines', '200,000', '12,000', '26,455,000'], ['33', 'Togo', '200,000', '12,000', '26,455,000'], ['34', 'Guinea', '160,000', '9,600', '21,164,000'], ['35', 'Yemen', '120,000', '7,200', '15,873,000'], ['36', 'Cuba', '100,000', '6,000', '13,227,000'], ['37', 'Panama', '100,000', '6,000', '13,227,000'], ['38', 'Bolivia', '90,000', '5,400', '11,904,000'], ['39', 'Timor Leste', '80,000', '4,800', '10,582,000'], ['40', 'Central African Republic', '65,000', '3,900', '8,598,000'], ['41', 'Nigeria', '40,000', '2,400', '5,291,000'], ['42', 'Ghana', '37,000', '2,220', '4,894,000'], ['43', 'Sierra Leone', '36,000', '2,160', '4,761,000'], ['44', 'Angola', '35,000', '2,100', '4,629,000'], ['45', 'Jamaica', '21,000', '1,260', '2,777,000'], ['46', 'Paraguay', '20,000', '1,200', '2,645,000'], ['47', 'Malawi', '16,000', '960', '2,116,000'], ['48', 'Trinidad and Tobago', '12,000', '720', '1,587,000'], ['49', 'Zimbabwe', '10,000', '600', '1,322,000'], ['50', 'Liberia', '6,000', '360', '793,000'], ['51', 'Zambia', '2,000', '120', '264,000'], ['52', 'United States','', '11,408', '25,150,000']]
df=pd.DataFrame(main_test)
df.drop(0,axis=1,inplace=True)
df


# In[23]:


df.to_csv('coffeedataset1.csv')


# In[68]:


url='https://m.imdb.com/chart/top/'
data=requests.get(url)
print(data)


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


plt.bar(df['1'],df['2'])

