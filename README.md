🎓 Student Depression Prediction – Uçtan Uca MLOps Projesi

Bu proje, öğrencilerde depresyon durumunu tahmin etmek için geliştirilmiş, uçtan uca bir MLOps tabanlı makine öğrenmesi sistemidir.

Projede; verinin alınmasından başlayarak model eğitimi, deney takibi, artefact yönetimi ve canlı ortama otomatik dağıtıma kadar tüm süreçler gerçek bir üretim ortamına uygun şekilde tasarlanmıştır.

🚀 Proje Amacı

Bu projenin amacı;

öğrencilerin akademik ve yaşam alışkanlıklarına ait özellikleri kullanarak:

öğrencinin depresyonda olup olmadığını tahmin eden bir makine öğrenmesi sistemi geliştirmektir.

Sınıflandırma problemi:

0 → Depresyonda değil

1 → Depresyonda

📊 Veri Seti

Model aşağıdaki özellikleri kullanır:

Age

Gender

Department

CGPA

Sleep duration

Study hours

Social media hours

Physical activity

Stress level

Hedef kolon: öğrencinin depresyon durumu.

🧠 Kullanılan Teknolojiler

Python

Scikit-learn

MLflow

DagsHub

Docker

Flask

GitHub Actions

AWS EC2

AWS ECR

AWS S3


🏗 Genel Mimari

Proje aşağıdaki uçtan uca pipeline yapısına sahiptir:

Data Ingestion
      ↓
Data Validation
      ↓
Data Transformation
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Artifact kayıtları (DagsHub + S3)


🔄 Veri Dönüştürme (Data Transformation)

Veri ön işleme süreci ColumnTransformer kullanılarak yapılmaktadır.

Sayısal değişkenler için:

Median imputation

StandardScaler

Kategorik değişkenler için:

Most frequent imputation

OneHotEncoder

Eğitilen preprocessing pipeline:

pickle olarak kaydedilir

inference sırasında tekrar kullanılır.

Dönüştürülmüş veri setleri:

train_transformed.csv

test_transformed.csv

olarak artefact klasörüne yazılır.

🤖 Model Eğitimi

Aynı veri üzerinde birden fazla model eğitilmekte ve en iyi model otomatik olarak seçilmektedir.

Eğitilen modeller:

RandomForestClassifier

AdaBoostClassifier

GradientBoostingClassifier

Her model için:

GridSearchCV ile hiperparametre araması yapılır

Test verisi üzerinde accuracy hesaplanır

En iyi accuracy değerine sahip model seçilir

Seçilen model:

pickle olarak kaydedilir

parametreleri ayrı bir JSON dosyasına yazılır.

📈 Model Değerlendirme ve Deney Takibi

Model değerlendirme aşamasında:

Accuracy

Confusion Matrix

hesaplanmaktadır.

Tüm deneyler MLflow ile takip edilir.

MLflow altyapısı DagsHub üzerinden çalışmaktadır.

Her run sırasında:

accuracy

confusion matrix

model adı

model hiperparametreleri

eğitilmiş model

MLflow’a loglanır.

☁️ Artifact Yönetimi

Bu projede artefact’lar iki farklı ortamda tutulur.

✔ DagsHub

MLflow experiment kayıtları

model artefact’ları

✔ AWS S3

Eğitim tamamlandıktan sonra:

artifacts/
klasörünün tamamı otomatik olarak S3 bucket’a senkronize edilir.


🌐 API Servisi (Flask)

Model servis tarafı Flask ile yazılmıştır.

Endpoint’ler
Ana sayfa

GET /
Tahmin formunu gösterir.

Tahmin

POST /predict

Formdan gelen veriler ile tahmin yapar.

Eğitim

GET /train

Tüm training pipeline’ını tetikler.

🐳 Docker

Uygulama Docker container içerisinde çalışmaktadır.

Flask uygulaması container içinde:

0.0.0.0:8080
portunda dinler.

🔁 CI / CD Süreci

GitHub Actions üzerinden 3 aşamalı bir pipeline kurulmuştur.

1️⃣ Continuous Integration

Repository checkout

Lint

Unit test adımları

2️⃣ Continuous Delivery

Docker image build edilir

AWS ECR’a push edilir

3️⃣ Continuous Deployment

AWS EC2 üzerinde çalışan self-hosted runner kullanılır

ECR’dan en güncel image çekilir

Container ayağa kaldırılır

☁️ Bulut Altyapısı

AWS EC2 → uygulamanın çalıştığı sunucu

GitHub self-hosted runner → EC2 üzerinde çalışır

AWS ECR → Docker image deposu

AWS S3 → tüm artifacts klasörü saklanır

🧪 Deney Takibi

Tüm eğitim süreçleri:

MLflow

DagsHub

üzerinden takip edilir.

Her eğitim çalışması için:

metrikler

parametreler

confusion matrix

model dosyası

kayıt altına alınır.

📁 Proje Klasör Yapısı



```text
student-depression-mlops-project/
│
├── artifacts/
│   ├── data/
│   ├── objects/
│   ├── models/
│   └── reports/
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_train.py
│   │   └── model_eval.py
│   │
│   ├── pipeline/
│   │   ├── datapipeline.py
│   │   └── modelpredictPipeline.py
│   │
│   ├── entity/
│   │   ├── artifacts_entity.py
│   │   └── config_entity.py
│   │
│   ├── cloud/
│   │   └── s3_syncer.py
│   │
│   └── utils/
│
├── templates/
│   └── index.html
│
├── app.py
├── main.py
├── Dockerfile
├── requirements.txt
│
└── .github/
    └── workflows/
        └── workflow.yml


▶️ Lokal Çalıştırma
Gerekli paketler

pip install -r requirements.txt


Eğitim pipeline’ını çalıştırmak

python main.py

API’yi çalıştırmak

python app.py

Tarayıcıdan:

http://localhost:8080  (deploy sonrası url:8080)


🔐 Ortam Değişkenleri

Aşağıdaki bilgiler GitHub Actions secrets olarak tanımlanmıştır:

AWS_ACCESS_KEY_ID

AWS_SECRET_ACCESS_KEY

AWS_REGION

ECR_REPOSITORY_NAME

⭐ Projenin Öne Çıkan Noktaları

Tamamen modüler MLOps pipeline mimarisi

Otomatik model seçimi (GridSearchCV)

MLflow + DagsHub ile deney takibi

Artifact’ların S3 üzerinde merkezi olarak tutulması

Docker ile containerlaştırılmış servis

GitHub Actions ile uçtan uca CI/CD

Self-hosted runner ile gerçek production benzeri deployment

👤 Geliştirici

Muharrem Aydoğan

Bu proje, gerçek hayattaki MLOps süreçlerini göstermek amacıyla;
model yaşam döngüsü, deney takibi ve üretim ortamına dağıtım adımlarını kapsayacak şekilde geliştirilmiştir.
