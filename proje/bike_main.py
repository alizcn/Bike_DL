#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import r2_score



warnings.filterwarnings("ignore")


# In[ ]:


data=pd.read_csv("Sales.csv")


# In[ ]:


#ilk 5 satır
data.head()


# In[ ]:


#Son 5 satır
data.tail()


# In[ ]:


#satır ve sütun sayıları
data.shape


# In[ ]:


#veri seti hakkında genel bilgi,veri tiplerinin alanlara göre çeşitli olduğu görünmektedir.
data.info()


# In[ ]:


data['Date'] = pd.to_datetime(data['Date'])


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.index


# In[ ]:


#sütunlar özelinde toplam kayıt sayıları,ortalama değerleri,standart sapma,min ve max değerler,çeyrek değerleri
data.describe().T


# In[ ]:


# Veriseti içerisinde boş değer var mı yok mu
data.isnull().values.any()


# In[ ]:


# Veriseti içerisinde boş değer var ise adedi
data.isnull().sum()


# In[ ]:


for i in data.columns:
    print("Column Name : ",i)
    print("\n*************\n")
    print(data[i].unique())
    print("\n*************\n")


# In[ ]:


# her sütuna ait verilerin toplam değerleri
for i in data.columns:
    print("Column Name : ",i)
    print("\n*************\n")
    print(data[i].value_counts())
    print("\n*************\n")


# In[ ]:


type(data["Country"].head())


# In[ ]:


data[["Country"]].head()


# In[ ]:


data.columns


# In[ ]:


#ülkelerin toplam siparişleri
data.groupby("Country")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#yaş gruplarına göre sipariş sayıları
data.groupby("Age_Group")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#ürünlere ait toplam siparişler
data.groupby("Product")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#ürünlere ait elde edilen kar
data.groupby("Product")["Profit"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#ürün kategori bazlı elde edilen karlar
data.groupby("Product_Category")["Profit"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#ürün kategori bazlı maliyetler
data.groupby("Product_Category")["Cost"].sum().sort_values(ascending=False).head(10)


# In[ ]:


data.groupby("State")["Profit"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#eyaletlere göre verilen siparişler
data.groupby("State")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#yıllara göre verilen sipariş sayıları toplamı
data.groupby("Year")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#yıllara göre elde edilen toplam kar
data.groupby("Year")["Profit"].sum().sort_values(ascending=False).head(10)


# In[ ]:


data.columns


# In[ ]:


data.groupby("Month")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#aylara göre toplam siparişler
data.groupby("Month")["Order_Quantity"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#aylara göre elde edilen toplam gelir.
data.groupby("Month")["Revenue"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#yıllara göre elde edilen toplam gelir.
data.groupby("Year")["Revenue"].sum().sort_values(ascending=False).head(10)


# In[ ]:


#cinsiyetlere göre yaş ortalaması ve sipariş sayıları
data.groupby("Customer_Gender").agg({"Customer_Age": ["mean"],"Order_Quantity": "mean"})


# In[ ]:


data.columns


# In[ ]:


#ülkelere göre yaş oralaması ve sipariş ortalamaları
data.groupby("Country").agg({"Customer_Age": ["mean"],"Order_Quantity": "mean"})


# In[ ]:


#ülkelere göre maliyetler,karlar,gelirler
data.groupby("Country").agg({"Revenue": "mean","Profit": "mean","Cost": "mean"})


# In[ ]:


#ürünlere göre maliyetler,karlar,gelirler
data.groupby("Product").agg({"Revenue": "mean","Profit": "mean","Cost": "mean"})


# In[ ]:


#ürün kategori bazlı göre maliyetler,karlar,gelirler
data.groupby("Product_Category").agg({"Revenue": "mean","Profit": "mean","Cost": "mean"})


# In[ ]:


#aylara göre ürünlere ait oplam karlar
pd.pivot_table(data, values='Profit', index='Product', columns='Month', aggfunc='sum')


# In[ ]:


#ürünlerin ay olarak toplam karları
pd.pivot_table(data, values='Profit', index='Product', columns='Month', aggfunc='sum')


# In[ ]:


#ürümleri ülkelere göre kar toplamları
pd.pivot_table(data, values='Profit', index='Product', columns='Country', aggfunc='sum')


# In[ ]:


yearly_sales = data.groupby('Year').agg({
    'Order_Quantity': 'sum',
    'Revenue': 'sum',
    'Profit':'sum'
}).reset_index()


# In[ ]:


sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='Blues')


# In[ ]:


plt.figure(figsize=(8, 4))
plt.plot(yearly_sales['Year'], yearly_sales['Order_Quantity'], marker='o', linestyle='-',color='green')
plt.title('Yıllık Toplam Satış Miktarı')
plt.xlabel('Yıl')
plt.ylabel('Toplam Sipariş Miktarı')
plt.grid(True)
plt.show()


# In[ ]:


plt.figure(figsize=(8,4))
plt.plot(yearly_sales['Year'], yearly_sales['Revenue'], marker='o', linestyle='-', color='green')
plt.title('Yıllık Toplam Satış Geliri')
plt.xlabel('Yıl')
plt.ylabel('Toplam Gelir')
plt.grid(True)
plt.show()


# In[ ]:


plt.plot(yearly_sales['Year'], yearly_sales['Profit'], marker='o', linestyle='-', color='green')
plt.title('Yıllık Toplam Kar Miktarı')
plt.xlabel('Yıl')
plt.ylabel('Toplam Kar Miktarı')
plt.grid(True)
plt.show()


# In[ ]:


#müşteri yaşlarına göre elde edilen kar
sns.lineplot(data=data, x="Customer_Age", y="Profit",ci=None,color="green")
plt.title('Satışların yaş gruplarına göre karı')
plt.grid(True)


# In[ ]:


#Ülkelere göre elde edilen karlar
sns.lineplot(data=data, x="Country", y="Profit",ci=None,color="green")
plt.title('Ülkelerin kar durumları')
plt.grid(True)


# In[ ]:


#ürün kategorilerinden elde edilen gelirler
sns.lineplot(data=data, x="Product_Category", y="Revenue",color="green")
plt.title('Ürün kategorilerine göre gelir')
plt.grid(True)


# In[ ]:


sns.displot(data=data, x="Customer_Age",bins=10,color="green")
plt.title('Yaş Dağılımı')


# In[ ]:


#Yaş grupları dağılımı
sns.displot(data=data, x="Age_Group",bins=30,color="green")
plt.title('Yaş grupları dağılımı')
plt.xticks(rotation=90)
plt.show()


# In[ ]:


#ay dağılımı
sns.displot(data=data, x="Month",bins=40,color="green")
plt.title("Satış Yapılan Ay Dağılımı")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.displot(data=data, x="Sub_Category",bins=40,color="green")
plt.title("Alt kategori dağılımı")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


labels=data["Country"].unique()
views=data.groupby("Country")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Ülke satışlarının gelir durumu')
plt.show()


# In[ ]:


labels=data["Age_Group"].unique()
views=data.groupby("Age_Group")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Yaş gruplarının satışlar üzerindeki gelir durumu')
plt.show()


# In[ ]:


labels=data["Year"].unique()
views=data.groupby("Year")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Yıllara göre gelir durumu')
plt.show()


# In[ ]:


labels=data["Month"].unique()
views=data.groupby("Month")["Revenue"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Aylara göre gelir durumu')
plt.show()


# In[ ]:


data.columns


# In[ ]:


labels=data["Customer_Gender"].unique()
views=data.groupby("Customer_Gender")["Order_Quantity"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Cinsiyetlerin sipariş durumları')
plt.show()


# In[ ]:


labels=data["Product_Category"].unique()
views=data.groupby("Product_Category")["Profit"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Kategoriye göre kar durumu')
plt.show()


# In[ ]:


labels=data["Product_Category"].unique()
views=data.groupby("Product_Category")["Order_Quantity"].sum()
plt.pie(views,labels=labels,autopct="%1.1f%%",shadow=True,wedgeprops={"width":0.3})
plt.title('Kategoriye göre sipariş durumu')
plt.show()


# In[ ]:


print(data["Month"].unique())
print(data["Age_Group"].unique())
print(data["Customer_Gender"].unique())
print(data["Country"].unique())
print(data["State"].unique())
print(data["Product_Category"].unique())
print(data["Sub_Category"].unique())


# In[ ]:


# Assuming 'data' is your DataFrame
label_encoder = LabelEncoder()
encode = ["Month", "Age_Group", "Customer_Gender", "Country", "State", "Product_Category","Product","Sub_Category"]

# Loop through each column in the 'encode' list
for column in encode:
    # Check if the column has 4 or fewer unique values
    if data[column].nunique() <= 4:
        # Apply label encoding if there are 4 or fewer unique values
        data[column] = label_encoder.fit_transform(data[column])
        print(f"Applied label encoding to: {column}")
    else:
        # Apply one-hot encoding if there are more than 4 unique values
        data = pd.get_dummies(data, columns=[column])
        print(f"Applied one-hot encoding to: {column}")

# Display the first few rows to verify the changes
print(data.head())


# In[ ]:


data.info()


# In[ ]:


remove = "Date"
data = data.drop(columns=remove)
print(data.head())


# In[ ]:


data


# In[ ]:


y = data['Revenue']
X = data.drop(columns=['Revenue'])


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[222]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


# Modeli oluştur
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Modeli derle
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğit
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=32, verbose=1)

# Modeli test seti üzerinde değerlendir
test_loss = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test).flatten()  # Tahminleri 1 boyutlu hale getirin

# RMSE ve R^2 hesapla
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
r2 = r2_score(y_test, y_pred)

print("Test Loss (MSE):", test_loss)
print("RMSE:", rmse)
print("R^2 Score:", r2)

# Eğitim ve doğrulama kayıplarını çizdir
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("Eğitim ve Doğrulama Kaybı")
plt.show()

# Tahmin edilen ve gerçek değerleri karşılaştır
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Gerçek Değerler")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Gerçek vs. Tahmin")
plt.show()

