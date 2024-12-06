# Python 3.8 imajını kullan
FROM python:3.8

# Çalışma dizinini ayarla
WORKDIR /app

# requirements.txt dosyasını kopyala ve kütüphaneleri yükle
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Klasörü konteynıra kopyala
COPY "proje" /app/myfolder

# Jupyter Notebook'u çalıştır
CMD ["jupyter-notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
