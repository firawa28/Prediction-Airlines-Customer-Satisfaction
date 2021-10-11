# Prediction-Airlines-Customer-Satisfaction

Pada repository ini merupakan pengembangan supervised machine learning yang berfungsi untuk memprediksi kepuasan pelanggan terhadap pelayanan yang diberikan oleh suatu maskapai penerbangan, yaitu Invistico Airlines. Tahapan yang dilakukan ketika membangun model ini adalah

1. Load Library
2. EDA
   - Data Exploration
   - Univariate Analysis
   - Bivariate Analysis
   - Multivariate Analysis
3. Data preprocessing dan Feature Engineering
4. Modelling
5. Evaluation Model
6. Business Recomendation

## Introduction
Invistico Airlines merupakan maskapai asal negara Indonesia yang selalu mengutamakan kualiatas, keamanan, dan pelayanan. Berdasarkan prinsip tersebur, pada tahun 2020, Invistico Airlines berhasil menempati TOP 5 World Airlines Awards. Namun seiring berjalannya waktu, pada tahun ini Invistico Airlines mengalami penurunan ranking menjadi posisi 12 World Airlines Awards. Hal ini didukung juga berdasarkan survey yang telah dilakukan oleh Tim Reasearch Invistco Airlines diketahui bahwa sebanyak 45% customer Invistico Airlines tidak puas terhadap pelayan yang diberikan. Hal ini tentunya perlu dilakukan evaluasi, karena mengakibatkan menurunnya Brand Image, Jumlah Pelanggan, dan Revenue yang dihasilkan. Oleh karena itu, management Invistico Airlines ingin mengetahui faktor-faktor apa saja yang mempengaruhi kepuasan pelanggan dan meningkatkan satisfaction rate Invistico Airlines.

## Goals

1. Mengetahui faktor-faktor apa aja yang berpengaruh besar dalam kepuasan pelanggan Invistico Airlines.
2. Meningkatkan Customer Satisfaction Rate Invistico Airlines.

## Data Understanding
Pada dataset [Invistico_Airlines.csv](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Invistico_Airline.csv) terdapat dua kategori Feature dan 1 target.

### Feature
1. Customer Profile
   - Age : Umur customer Invistico Airlines.
   - Gender : Jenis Kelamin Invistico Airlines.
   - Customer Type : Jenis Customer Invistico Airlines. Loyal Customer atau Disloyal Customer.
   - Type OF Travel : Jenis Penerbangan yang digunakan oleh customer Invistico Airlines. Personal Travel atau Business Travel.
   - Class : Jenis Cabin yang dibooking oleh customer Invistico Airlines. Economy, Economy Plus, atau Businness.
   - Flight Distance : Jarak Penerbangan yang ditempuh oleh customer Invistico Airlines.
   
2. Services
   - Seat comfort : Rating service dari tingkat kenyamanan kursi maskapai Invistico Airlines.
   - Departure/Arrival time convenient : Rating service dari tingkat kenyamanan pada waktu keberangkatan/kedatangan.
   - Food and drink : Rating service dari makanan dan minuman yang disajikan oleh Inflight wifi service.
   - Gate location : Rating service dari lokasi pintu pesawat di mana customer naik ke maskapai Invistico Airlines..
   - Inflight wifi service : Rating service dari layanan wifi di dalam maskapai Invistico Airlines. 
   - Inflight entertainment : Rating service dari layanan entertainment di dalam maskapai Invistico Airlines.
   - Online support : Rating service dari pelayanan online yang diberikan oleh maskapai Invistico Airlines.
   - Ease of Online booking : Rating service dari kemudahan untuk melakukan booking online maskapai Invistico Airlines.
   - On-board service : Rating service dari pelayanan yang diberikan di dalam maskapai Invistico airlines.
   - Leg room service : Rating service dari kualitas ruang kaki maskapai Invistico Airlines.
   - Baggage handling : Rating service dari kualitas pelayanan bagasi maskapai Invistico Airlines. 
   - Checkin service : Rating service dari kemudahan check-in maskapai Invistico Airlines.
   - Cleanliness : Rating service dari tingkat kebersihan maskapai Invistico Airlines.
   - Online boarding : Rating service dari layanan boarding online maskapai Invistico airlines.
   - Departure Delay in Minutes : Waktu keterlambatan melakukan penerbangan.
   - Arrival Delay in Minutes : Waktu keterlambatan ketika landing di tujuan.

### Target
- Satisfaction : Kepuasan customer maskapai Invistico Airlines. Satissfied atau Dissatisfied.

## Exploratory Data Analysis (EDA)
1. Karaktistik customer Invistico Airlines secara umum merupakan seorang business man yang ber umur 27 – 51.
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Characteristics%20of%20Customer%20Invistico%20Airlines.PNG "Characteristics of Customer Invistico Airlines")
2. Cabin class economy dan economy plus memiliki tingkat ketidakpuasan yang tinggi dibandingkan dengan bisnis cabin class. 
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Satisfaction%20Rate%20of%20Cabin%20Class.png "Satisfaction per Cabin Class Invistico Airlines")
3. Invistico airlines membagi menjadi 2 kategori untuk level rating kepuasan. Nilai rating 0-2 dinilai customer tidak puas terhadap pelayanan yang diberikan, nilai rating 3-5 dinilai customer puas terhadap pelayanan yang diberikan.
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Level%20of%20Rating.PNG "Level of Rating")
4. Service yang memiliki rating di bawah 3 adalah Departure/Arrival time convenient, Gate Location, Food and Drink, dan Seat Comfort.
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Average%20of%20Feature%20Ratings.png "Average of Feature Ratings")
5. Terdapat anomaly pada feature-feature yang memiliki rating di bawah 3, karena tidak ada kecenderungan pada feature-feature tersebut apakah menunujukan satisfied dan dissatisfied. Dalam artian, ada customer yang dissatisfied tetapi tetap memberikan nilai rating tinggi, dan ada customer yang satisfied tetapi memberikan nilai rating rendah.
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Anomaly%20Data%20in%204%20Feature.png "Anomaly Feature Rating")

Berdasarkan analisa EDA di atas, karena terdapat anomaly data maka diputuskan untuk melakukan analisa lebih dalam menggunakan model machine learning untuk mengetahui feature yang berperan besar dalam mempengaruhi satisfaction customer Invistico Airlines.


## Data Preprocessing
Sebelum kita melakukan modelling machine learning, tahapan yang harus dilakukan adalah Data Preprocessing. Tujuanya adalah untuk mendapatkan hasil model prediksi yang baik. Berikut merupakan tahapan-tahapan yang dilakukan:

1. Handle Missing Value <br>
   Diketahui bahwa terdapat missing value pada fiture Arrival Delay in Minutes. Untuk handle missing value tersebut, kami handle dengan cara filling missing value dengan nilai mediannya.
2. Handle Feature Categorical <br>
   Ada dua metode yang kami lakukan, yaitu Label Encoding dan One-Hot Encoding <br>
a.	Label Encoding dilakukan pada feature Class dan target Satisfaction <br>
b.	One-Hot Encoding dilakukan pada feature Gender, Customer Type dan Type of Travel <br>
 
3. Feature Selection <br>
   Pada tahapan ini, kami hanya memilih feature-feature memiliki nilai korelasi > |0.1| dengan target. <br>
   
4. Split Data <br>
   Pada tahapan ini, kami melakukan split data untuk training set dan testing set dengan proporsi perbandingan 70:30. 70 untuk training set dan 30 untuk testing set. Selain itu kami menggunakan random_state = 42.

## Modelling dan Evaluation Model.

### Machine Learning Modelling

Karena kita memiliki target seimbang, dan kita ingin meningkatkan kualitas prediksi, yaitu dengan **meminimalisir false positif** yang terjadi dimana false positif terjadi ketika Actual "No" atau dissatisfied namun model memprediksi "Yes" atau Satisfied. Oleh karena itu, kita akan menggunakan evaluation score **Precision**. Selanjutnya model yang digunakan untuk training dan testing adalah

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Adaboost Classifier
5. Gradient Boosting Classifier

### Model Evaluation

Random Forest Classifier: <br>
| Evaluation Score | Training | Testing |
| --- | --- | --- |
| Akurasi | 100% |  95% |
| Precision | 100% | 97% |
| Recall | 100% | 95% |
| F1-Score | 100% | 96% |

Berdasarkan hasil evaluation score yang telah dilakukan dari beberapa model, diketahui bahwa model Random Forest Classifier yang memiliki tingkat Precision tertinggi, yaitu sebesar 97%. Selain itu, dapat dilihat bahwa model ini memiliki rentang evaluation score Precision pada training dan testing score yang kecil, sehingga dapat kita simpulkan model ini sangat baik karena tidak terjadi underfitting maupun overfitting. Oleh karena itu, dipilih model Random Forest Classifier untuk melakukan prediksi satisfaction customer maskapai Invistico Airlines.

## Feature Importance
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Feature%20Importance%20Score.png "Feature Importance by Model Random Forest Classifier")

Berdasarkan feature importance menurut model Random Forest Classifier feature-feature yang paling berperan besar dalam mempengaruhi satisfaction customer maskapai Invistico Airlines adalah:
1. Inflight Entertaiment
2. Seat comfort
3. Online Support
4. Ease of Online Booking

Dengan kita mengetahui feature yang berperan besar dalam memngaruhi satisfaction customer, tentunya kita dapat melakukan improvement pada feature tersebut agar tingkat satisfaction invistico airline dapat meningkat.

## Business Recommendation
Dalam merekomendasikan bisnis, kami membagi menjadi 4 bagian utama yaitu:
1.	Action/Process
2.	Area Improvement
3.	Mechanisme
4.	Impact

Berikut merupakan ringkasan dari Business Recomendation yang kami usulkan untuk setiap 4 feature yang paling berpengaruh besar dalam mempengaruhi satisfaction customer:
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Business%20Recomendation.PNG "Busniess Recomendation")

Kami ketahui bahwa faktor utama yang mempengaruhi tingkat satisfaction yang pertama adalah **Inflight Entertainment**. Kami mencoba mereview terhadap kesisteman yang ada di dalam sistem kami. Kemudian kami juga mencoba melakukan komparasi dengan competitor. Dari hasil tersebut, kami menemukan bahwa kami bisa melakukan improvement dari sisi hardware, sisi software, dan konten management. Mekanisme yang kami lakukan dari hal tersebut  adalah kami melakukan refreshment terhadap Inflight Entertainment kami dan tentunya di sini ada impact cost yang kami akan lakukan perhitungan nya di section berikutnya. Saat ini kami melihat bahwa Impactnya terhadap satu orang customer kami akan dibebankan sebesar 5 usd pada harga tiketnya dan Namun hal tersebut akan di cover sementara oleh Invistico Airlines.

Selanjutnya dari sisi **Seat Comfort**, yang kami lihat adalah mengenai jarak antara kursi dan sandaran kepala. Namun untuk melakukan improvement dari fitur ini, kami harus mengevaluasi juga dari jumlah cabin dan jumlah penumpang di dalam pesawat sehingga untuk feature seat comfort  akan evaluasi lebih lanjut. 

Kemudian yang ke-3 mengenai **Online Support**. Dalam hal ini kami melihat ada beberapa hal yang perlu kami perbaiki dari sisi customer support kami. Berdasarkan evaluasi terhadap KPI performa customer support, kami perlu melakukan re-training terhadap semua customer support kami, sehingga kami bisa melihat bahwa apa yang dilakukan oleh customer support memang sudah dibutuhkan dengan apa yang di minta oleh pelanggan. Improvemrnt pada online support tidak ada impact terhadap cost karena kami mempunyai trainner dari sisi internal kami.

Terakhir adalah **Ease of Online booking** dalam hal ini yang pertama kami lakukan adalah melakukan review terhadap contract travel agent. Setelah kami melakukan review, ternyata ada travel agent yang contractnya sudah putus, namun travel agent tersebut mempunyai renewal terhadap proses bookingnya. Kemudian yang kedua kami melakukan perbaikan terhadap dari sisi UI/UX seperti mobile application, website dan travel agent. Karena UI/UX sudah satu paket dengan Inflight Entertainment kami, maka cost untuk improvement sudah satu bundle dengan Inflight Entertainment. 


Langkah selanjutnya adalah kami melakukan simulasi perhitungan sesuai bisnis yang kami rekomendasikan.
![alt text](https://github.com/firawa28/Prediction-Airlines-Customer-Satisfaction/blob/main/Gambar/Business%20Simulation.PNG "Busniess Simulation")

Kami asumsikan kami mempunyai 130.000 pelanggan, dan kepadatan setiap penerbangan sekitar 80 sampai 90% kursi. Kita ketahui bahwa pada Q1 -  Q3 2021 terjadi penurunan revenue dari 21,8 juta usd menjadi 20,3 juta usd dan saat kami lakukan survey pada Q4 2021 kami menemukan bahwa tingkat satisfaction berada pada angka 55%. Berdasarkan hal tersebut kami mengusulkan salah satunya launching new Inflight Entertaiment dan hal ini tentunya membutuhkan cost. Cost yang kami keluarkan sesuai dengan jumlah customer, dengan prediksi Invistico Airlines butuh cost investment rata-rata sekitar 1,23 juta dolar untuk setiap kuarter. Setelah launching new Inflight Entertaiment, tentunya tidak langsung signifikan revenue kita akan naik. Oleh karena itu, pada Q2 2022, Q4 2022, Q2 2023 dan Q4 2023 kami rutin untuk melakukan survey untuk evaluasi.

Kami mengasumsikan bahwa setelah beberapa improvement yang kami lakukan bahwa tingkat kepuasan pada Q2 2022 ada kenaikan pada angka 58%. Dari sini, kita bisa mulai menaikan harga tiket yang tentunya masih dalam rentang batas atas dan batas bawah governance sehingga tidak akan menajdi konflik di public nantinya. Sesuai dengan rekomendasi, yang tadinya 5 USD dibebankan pada perusahaan sekarang 5 USD dibebankan per pelanggan. Selanjutnya kami juga mengusulkan untuk melakukan kerja sama dengan beberapa perusahan untuk memasang iklan di maskapai kami yang tentunya akan menjadi new revenue bagi kami.

Kesimpulannya, walapun kita mengalami penurunan revenue pada 2022 sebesar -11 % atau sekitar 8.6 juta usd. Namun pada tahun 2023 kami akan mendapatkan kenaikan revenue sebesar 19% atau sekitar 13.9 USD dan mencapai tingkat kepuasan pelanggan sebesar 75 %.


# Team Behind This Project
* As the final project to finish learning path for Data Science in [Rakamin Academy](https://rakamin.com/), this model was built from a growth team called **Hakuna Matata**<br>

**The Team Member**<br>
| Member |LinkedIn url |
| --- | --- |
| **Our Friendly Mentor: Muhammad Mirza Fahmi** | https://www.linkedin.com/in/mmirzafahmi/ |
| **Fikri Rafhsanjani Wahiyat (me)** | https://www.linkedin.com/in/fikri-rafhsanjani/ |
| **Asep Hermawan** | https://www.linkedin.com/in/asep-hermawan-9052621a/|
| **Fitri Nabilah Asyharti** | https://www.linkedin.com/in/fitri-nabilah-asyharti-772403175/ |
| **Syifa Aulia** | - |
| **Rizki Dwi Santoso** | https://www.linkedin.com/in/rizki-dwi-santoso/ |
