# 📘 Laporan Proyek Machine Learning - *Tema Anggara*

## 🌐 Domain Proyek: Klasifikasi Risiko Banjir

### 📌 Latar Belakang

Banjir merupakan bencana alam yang sering terjadi di banyak wilayah, menyebabkan kerugian ekonomi, kerusakan infrastruktur, dan bahkan korban jiwa. Dalam menghadapi risiko ini, pemetaan potensi banjir berdasarkan data numerik sangat penting sebagai dasar mitigasi.

Teknologi Machine Learning dapat membantu mengklasifikasikan tingkat risiko banjir secara otomatis dari data numerik seperti curah hujan, topografi, tata guna lahan, dan kondisi drainase. Proyek ini bertujuan untuk membangun model klasifikasi risiko banjir berbasis data fitur lingkungan.

---

## 🎯 Business Understanding

### Problem Statements

1. Bagaimana memetakan wilayah ke dalam level risiko banjir rendah, sedang, dan tinggi berdasarkan data numerik?
2. Bagaimana mengetahui apakah fitur-fitur yang tersedia sudah cukup merepresentasikan target risiko banjir?

### Goals

1. Membagi wilayah ke dalam tiga tingkat risiko banjir secara otomatis berdasarkan `FloodProbability`.
2. Menguji apakah klasifikasi dapat dilakukan secara akurat tanpa perlu pelatihan model kompleks karena target merupakan kombinasi linier dari fitur.

### Solution Statement

* Menggunakan dua algoritma:

  * **Logistic Regression** sebagai baseline model.
  * **Random Forest Classifier** sebagai model kompleks berbasis ensemble.
* Melakukan evaluasi performa kedua model menggunakan metrik klasifikasi.

---
Berikut adalah bagian laporan **📊 Data Understanding** yang telah diperbarui dan dilengkapi dengan informasi fitur serta tautan dataset:

---

## 📊 Data Understanding

Dataset yang digunakan pada proyek ini adalah *Flood Prediction Dataset* yang bersumber dari platform Kaggle.

* **Link dataset**: [https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset]

Dataset ini berisi 21 kolom numerik yang merepresentasikan faktor-faktor lingkungan, sosial, dan infrastruktur yang dapat mempengaruhi risiko banjir pada suatu wilayah. Data berasal dari simulasi atau penggabungan beberapa sumber terpercaya seperti data iklim, topografi, dan infrastruktur wilayah.

### 📄 Fitur dalam Dataset

Berikut adalah deskripsi setiap fitur yang terdapat pada dataset:

1. **MonsoonIntensity**: Intensitas curah hujan saat musim hujan. Curah hujan tinggi meningkatkan risiko banjir.
2. **TopographyDrainage**: Kapasitas drainase berdasarkan topografi wilayah. Topografi yang baik dapat mengurangi genangan air.
3. **RiverManagement**: Efektivitas pengelolaan sungai, seperti pengerukan dan pemeliharaan tanggul.
4. **Deforestation**: Luas deforestasi di wilayah tersebut. Deforestasi meningkatkan limpasan air permukaan.
5. **Urbanization**: Tingkat urbanisasi. Wilayah urban cenderung memiliki permukaan tidak menyerap air seperti aspal dan beton.
6. **ClimateChange**: Dampak perubahan iklim yang menyebabkan curah hujan ekstrem atau tidak terduga.
7. **DamsQuality**: Kualitas dan pemeliharaan bendungan. Bendungan yang baik dapat mencegah banjir besar.
8. **Siltation**: Tingkat pengendapan sedimen di sungai dan waduk, yang dapat menurunkan kapasitas drainase alami.
9. **AgriculturalPractices**: Pola pertanian dan keberlanjutannya. Praktik pertanian yang buruk dapat memperparah risiko banjir.
10. **Encroachments**: Tingkat pelanggaran pembangunan di dataran banjir dan alur sungai.
11. **IneffectiveDisasterPreparedness**: Kurangnya rencana darurat, sistem peringatan dini, atau latihan bencana.
12. **DrainageSystems**: Kualitas sistem drainase wilayah, termasuk ukurannya dan status pemeliharaan.
13. **CoastalVulnerability**: Kerentanan daerah pesisir terhadap kenaikan air laut dan badai.
14. **Landslides**: Potensi tanah longsor akibat lereng curam atau tanah tidak stabil.
15. **Watersheds**: Jumlah dan karakteristik daerah aliran sungai.
16. **DeterioratingInfrastructure**: Infrastruktur yang rusak atau tersumbat seperti gorong-gorong dan saluran air.
17. **PopulationScore**: Tingkat kepadatan populasi yang memperbesar dampak dari banjir.
18. **WetlandLoss**: Hilangnya lahan basah yang berfungsi sebagai penyangga air alami.
19. **InadequatePlanning**: Perencanaan tata kota yang mengabaikan risiko banjir.
20. **PoliticalFactors**: Faktor politik seperti korupsi dan kurangnya dukungan anggaran untuk infrastruktur banjir.
21. **FloodProbability**: Probabilitas banjir secara keseluruhan di wilayah tersebut. Ini adalah **target variabel** dalam proyek klasifikasi ini.

### 🔍 Exploratory Data Analysis (EDA)

Untuk memahami data lebih dalam, dilakukan beberapa tahap:

* Menampilkan informasi umum (jumlah data, tipe data, nilai null, duplikat)
* Visualisasi distribusi target (`FloodProbability`)
* Analisis korelasi antar fitur
* Uji regresi linier untuk melihat apakah `FloodProbability` adalah kombinasi linier dari fitur lainnya

Temuan utama:

* Jumlah data lengkap seperti deskripsi sumber (50000, 21), tipe data int/float (kuantitatif), tidak ada nilai null dan duplikat)
* `FloodProbability` memiliki **korelasi tinggi secara kolektif** terhadap semua fitur.
* Hasil regresi linier menunjukkan bahwa `FloodProbability` bisa diprediksi **sempurna**, dengan R² sama dengan 1.
* Karena kecenderungan central, `FloodProbability` dapat **dibinning** menjadi kelas risiko (`FloodRiskLevel`) menggunakan teknik Quantile Binning.

---

## 🧹 Data Preparation

Langkah-langkah persiapan data:

1. **Quantile Binning**:

   * Membagi `FloodProbability` ke dalam 3 kelas risiko (`Low`, `Medium`, `High`) menggunakan `pd.qcut()`.
   * Tujuannya adalah agar distribusi kelas seimbang (equal frequency binning).

2. **Encoding Label**:

   * Mengubah label kategorikal ke numerik menggunakan `LabelEncoder`.

3. **Train-Test Split**:

   * Membagi data menjadi data latih dan uji sebanyak 70:30.

📌 **Catatan**: Tidak dilakukan normalisasi karena:

* Semua fitur sudah dalam skala yang relevan.
* Random Forest tidak sensitif terhadap skala fitur.

---

## 🤖 Modeling

### Algoritma 1: Logistic Regression

* Sederhana dan cepat.
* Cocok sebagai baseline untuk klasifikasi multi-kelas.
* Parameter utama: `max_iter=1000` agar konvergen.

### Algoritma 2: Random Forest Classifier

* Model ensemble berbasis pohon keputusan.
* Tidak sensitif terhadap multikolinearitas atau skala fitur.
* Parameter utama: `n_estimators=100`, `random_state=42`.

### 📈 Pemilihan Model Terbaik

Dibandingkan menggunakan metrik F1 Score dan akurasi. Random Forest diharapkan menangani kompleksitas dan interaksi antar fitur dengan lebih baik dibanding Logistic Regression.

---

## 🧪 Evaluation

### Metrik Evaluasi

Dalam proyek ini, kita menggunakan beberapa metrik klasifikasi yang umum:

* **Precision**: Kemampuan model untuk tidak mengklasifikasikan negatif sebagai positif.
* **Recall**: Kemampuan model untuk menangkap semua kasus positif sebenarnya.
* **F1-score**: Rata-rata harmonik precision dan recall, cocok saat data tidak seimbang.
* **Accuracy**: Persentase prediksi yang benar terhadap seluruh prediksi.

$$
F1 = 2 \cdot \frac{precision \cdot recall}{precision + recall}
$$

$$
Accuracy = \frac{TP + TN}{Total\ Observasi}
$$

### 📊 Hasil Evaluasi

#### 📌 Evaluation: Logistic Regression

```
              precision    recall  f1-score   support

        High       1.00      1.00      1.00      4845
         Low       1.00      1.00      1.00      4941
      Medium       1.00      1.00      1.00      5214

    accuracy                           1.00     15000
   macro avg       1.00      1.00      1.00     15000
weighted avg       1.00      1.00      1.00     15000
```

📍 **Interpretasi**:

* Logistic Regression menunjukkan akurasi sempurna (100%).
* Kemungkinan besar model ini **overfitting** atau **tidak mengalami generalisasi nyata**, karena kita sebelumnya menyimpulkan bahwa `FloodProbability` adalah kombinasi linier dari fitur-fitur.

#### 📌 Evaluation: Random Forest Classifier

```
              precision    recall  f1-score   support

        High       0.85      0.78      0.82      4845
         Low       0.80      0.86      0.83      4941
      Medium       0.67      0.67      0.67      5214

    accuracy                           0.77     15000
   macro avg       0.77      0.77      0.77     15000
weighted avg       0.77      0.77      0.77     15000
```

📍 **Interpretasi**:

* Random Forest menunjukkan hasil yang **lebih realistis** dibanding Logistic Regression.
* Performa terbaik ditemukan pada kelas **Low** dan **High**, namun performa pada kelas **Medium** masih bisa ditingkatkan.
* **Macro average** dan **weighted average** F1 berada di kisaran 0.77, mencerminkan model cukup baik secara keseluruhan.

### 📌 Kesimpulan Evaluasi

* Logistic Regression mencapai akurasi 100% karena target (`FloodProbability`) adalah kombinasi linier fitur — hal ini membuat klasifikasi menjadi sangat mudah, dan tidak mewakili generalisasi yang valid.
* Random Forest memberikan estimasi risiko yang lebih realistis dan **dapat digunakan dalam kondisi data yang lebih kompleks** di masa mendatang.
* Untuk implementasi nyata, Random Forest lebih direkomendasikan karena lebih tangguh terhadap noise dan tidak hanya mengandalkan hubungan linier.

---

