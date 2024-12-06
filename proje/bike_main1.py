#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


# In[2]:


data=pd.read_csv("Sales.csv")


# In[3]:


#ilk 5 satır
data.head()


# In[4]:


#Son 5 satır
data.tail()


# In[5]:


#satır ve sütun sayıları
data.shape


# In[6]:


#veri seti hakkında genel bilgi,veri tiplerinin alanlara göre çeşitli olduğu görünmektedir.
data.info()


# In[7]:


data['Date'] = pd.to_datetime(data['Date'])


# In[8]:


data.info()


# In[9]:


data.columns


# In[10]:


data.index


# In[11]:


#sütunlar özelinde toplam kayıt sayıları,ortalama değerleri,standart sapma,min ve max değerler,çeyrek değerleri
data.describe().T


# In[12]:


# Veriseti içerisinde boş değer var mı yok mu
data.isnull().values.any()


# In[13]:


# Veriseti içerisinde boş değer var ise adedi
data.isnull().sum()


# In[14]:


for i in data.columns:
    print("Column Name : ",i)
    print("\n*************\n")
    print(data[i].unique())
    print("\n*************\n")


# In[15]:


# her sütuna ait verilerin toplam değerleri
for i in data.columns:
    print("Column Name : ",i)
    print("\n*************\n")
    print(data[i].value_counts())
    print("\n*************\n")


# In[16]:


type(data["Country"].head())


# In[17]:


data[["Country"]].head()


# In[18]:


data.columns


# In[19]:


#ülkelerin toplam siparişleri
data.groupby("Country")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[20]:


#yaş gruplarına göre sipariş sayıları
data.groupby("Age_Group")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[21]:


#ürünlere ait toplam siparişler
data.groupby("Product")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[22]:


#ürünlere ait elde edilen kar
data.groupby("Product")["Profit"].sum().sort_values(ascending=False).head(10)


# In[23]:


#ürün kategori bazlı elde edilen karlar
data.groupby("Product_Category")["Profit"].sum().sort_values(ascending=False).head(10)


# In[24]:


#ürün kategori bazlı maliyetler
data.groupby("Product_Category")["Cost"].sum().sort_values(ascending=False).head(10)


# In[25]:


data.groupby("State")["Profit"].sum().sort_values(ascending=False).head(10)


# In[26]:


#eyaletlere göre verilen siparişler
data.groupby("State")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[27]:


#yıllara göre verilen sipariş sayıları toplamı
data.groupby("Year")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[28]:


#yıllara göre elde edilen toplam kar
data.groupby("Year")["Profit"].sum().sort_values(ascending=False).head(10)


# In[29]:


data.columns


# In[30]:


data.groupby("Month")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[31]:


#aylara göre toplam siparişler
data.groupby("Month")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[32]:


#aylara göre elde edilen toplam gelir.
data.groupby("Month")["Revenue"].sum().sort_values(ascending=False).head(10)


# In[33]:


#yıllara göre elde edilen toplam gelir.
data.groupby("Year")["Revenue"].sum().sort_values(ascending=False).head(10)


# In[34]:


#cinsiyetlere göre yaş ortalaması ve sipariş sayıları
data.groupby("Customer_Gender").agg({"Customer_Age": ["mean"],"Order_Quantity": "mean"})


# In[35]:


data.columns


# In[36]:


#ülkelere göre yaş oralaması ve sipariş ortalamaları
data.groupby("Country").agg({"Customer_Age": ["mean"],"Order_Quantity": "mean"})


# In[37]:


#ülkelere göre maliyetler,karlar,gelirler
data.groupby("Country").agg({"Revenue": "mean","Profit": "mean","Cost": "mean"})


# In[38]:


#ürünlere göre maliyetler,karlar,gelirler
data.groupby("Product").agg({"Revenue": "mean","Profit": "mean","Cost": "mean"})


# In[39]:


#ürün kategori bazlı göre maliyetler,karlar,gelirler
data.groupby("Product_Category").agg({"Revenue": "mean","Profit": "mean","Cost": "mean"})


# In[40]:


#aylara göre ürünlere ait oplam karlar
pd.pivot_table(data, values='Profit', index='Product', columns='Month', aggfunc='sum')


# In[41]:


#ürünlerin ay olarak toplam karları
pd.pivot_table(data, values='Profit', index='Product', columns='Month', aggfunc='sum')


# In[42]:


#ürümleri ülkelere göre kar toplamları
pd.pivot_table(data, values='Profit', index='Product', columns='Country', aggfunc='sum')


# In[43]:


yearly_sales = data.groupby('Year').agg({
    'Order_Quantity': 'sum',
    'Revenue': 'sum',
    'Profit':'sum'
}).reset_index()


# In[44]:


sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='Blues')


# In[45]:


plt.figure(figsize=(8, 4))
plt.plot(yearly_sales['Year'], yearly_sales['Order_Quantity'], marker='o', linestyle='-',color='green')
plt.title('Yıllık Toplam Satış Miktarı')
plt.xlabel('Yıl')
plt.ylabel('Toplam Sipariş Miktarı')
plt.grid(True)
plt.show()


# In[46]:


plt.figure(figsize=(8,4))
plt.plot(yearly_sales['Year'], yearly_sales['Revenue'], marker='o', linestyle='-', color='green')
plt.title('Yıllık Toplam Satış Geliri')
plt.xlabel('Yıl')
plt.ylabel('Toplam Gelir')
plt.grid(True)
plt.show()


# In[47]:


plt.plot(yearly_sales['Year'], yearly_sales['Profit'], marker='o', linestyle='-', color='green')
plt.title('Yıllık Toplam Kar Miktarı')
plt.xlabel('Yıl')
plt.ylabel('Toplam Kar Miktarı')
plt.grid(True)
plt.show()


# In[48]:


#müşteri yaşlarına göre elde edilen kar
sns.lineplot(data=data, x="Customer_Age", y="Profit",ci=None,color="green")
plt.title('Satışların yaş gruplarına göre karı')
plt.grid(True)


# In[49]:


#Ülkelere göre elde edilen karlar
sns.lineplot(data=data, x="Country", y="Profit",ci=None,color="green")
plt.title('Ülkelerin kar durumları')
plt.grid(True)


# In[50]:


#ürün kategorilerinden elde edilen gelirler
sns.lineplot(data=data, x="Product_Category", y="Revenue",color="green")
plt.title('Ürün kategorilerine göre gelir')
plt.grid(True)


# In[51]:


sns.displot(data=data, x="Customer_Age",bins=10,color="green")
plt.title('Yaş Dağılımı')


# In[52]:


#Yaş grupları dağılımı
sns.displot(data=data, x="Age_Group",bins=30,color="green")
plt.title('Yaş grupları dağılımı')
plt.xticks(rotation=90)
plt.show()


# In[53]:


#ay dağılımı
sns.displot(data=data, x="Month",bins=40,color="green")
plt.title("Satış Yapılan Ay Dağılımı")
plt.xticks(rotation=90)
plt.show()


# In[54]:


sns.displot(data=data, x="Sub_Category",bins=40,color="green")
plt.title("Alt kategori dağılımı")
plt.xticks(rotation=90)
plt.show()


# In[55]:


labels=data["Country"].unique()
views=data.groupby("Country")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Ülke satışlarının gelir durumu')
plt.show()


# In[56]:


labels=data["Age_Group"].unique()
views=data.groupby("Age_Group")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Yaş gruplarının satışlar üzerindeki gelir durumu')
plt.show()


# In[57]:


labels=data["Year"].unique()
views=data.groupby("Year")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Yıllara göre gelir durumu')
plt.show()


# In[58]:


labels=data["Month"].unique()
views=data.groupby("Month")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Aylara göre gelir durumu')
plt.show()


# In[59]:


data.columns


# In[60]:


labels=data["Customer_Gender"].unique()
views=data.groupby("Customer_Gender")["Order_Quantity"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Cinsiyetlerin sipariş durumları')
plt.show()


# In[61]:


labels=data["Product_Category"].unique()
views=data.groupby("Product_Category")["Profit"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Kategoriye göre kar durumu')
plt.show()


# In[62]:


labels=data["Product_Category"].unique()
views=data.groupby("Product_Category")["Order_Quantity"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Kategoriye göre sipariş durumu')
plt.show()


# In[ ]:




