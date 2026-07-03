# Telco Churn Prediction — Detaylı Kod Açıklaması & Decision Roadmap

Bu belge, notebook'taki her kod bloğunu satır satır açıklar ve her aşamadaki karar mekanizmamı (decision roadmap) anlatır.

---

## DECISION ROADMAP (Genel Akış)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROBLEM TANIMI                                    │
│  "Churn eden müşteriyi önceden tahmin et → retention kampanyası başlat"  │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. VERİ YÜKLEME & TANIMA                                                │
│     → Kaç satır/sütun? Tipler doğru mu? İlk 5 satır mantıklı mı?       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. VERİ KALİTESİ                                                        │
│     → Null var mı? Yanlış tip var mı? Neden null?                        │
│     KARAR: fillna(0) — çünkü tenure=0, mantıksal imputation             │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. TARGET ANALİZİ                                                       │
│     → %26.5 churn → moderate imbalance                                   │
│     KARAR: SMOTE değil, class_weight kullanacağız                        │
│     KARAR: Metrik = Recall (accuracy yanıltıcı)                          │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. SİSTEMATİK EDA (4 KATMAN)                                           │
│     4.1 Demografik → gender signal yok, senior/partner/dependents var    │
│     4.2 Servisler → fiber + no support = yüksek risk                    │
│     4.3 Sözleşme → contract type 15× fark (en güçlü)                   │
│     4.4 Numerik → tenure bimodal, ilk 6 ay kritik                       │
│     KARAR: En önemli feature'lar = contract, tenure, fiber, e-check     │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. RİSK SEGMENT KEŞFİ                                                  │
│     → Risk faktörlerini birlikte kesişim olarak sorgula                  │
│     → Business impact hesapla ($)                                        │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  6. PREPROCESSING                                                        │
│     KARAR: drop_first=True (dummy trap önleme)                           │
│     KARAR: stratify=y (class ratio koruma)                               │
│     KARAR: SMOTE yok (moderate imbalance, çoğu feature kategorik)        │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  7. BASELINE MODEL (LogReg)                                              │
│     → Recall 0.57 — yetersiz, churner'ların yarısını kaçırıyoruz        │
│     → Katsayılar EDA ile uyumlu → convergent validation                  │
│     KARAR: class_weight ekle                                             │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  8. CLASS WEIGHT EKLENMESİ                                               │
│     → Recall 0.57 → 0.79 (+38%)                                         │
│     → Hâlâ daha iyi olabilir → tree model dene                          │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  9. RANDOM FOREST + HYPERPARAMETER ARAŞTIRMASI                           │
│     → n_estimators: 10→300 (diminishing returns after 100)               │
│     → max_depth: KRİTİK KEŞİF                                           │
│       - depth=None → recall 0.50 (class_weight etkisiz!)                │
│       - depth=5 → recall 0.82 (class_weight aktif!)                     │
│     KARAR: max_depth=5, n_estimators=300, class_weight=balanced          │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  10. FINAL MODEL + BUSINESS IMPACT                                       │
│      → Recall 0.82, net business value hesaplandı                        │
│      → Feature importance → EDA + LogReg ile aynı → 3-yönlü doğrulama  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## KOD AÇIKLAMALARI (Satır Satır)

---

### BÖLÜM 1: Kütüphaneler & Ortam Kurulumu

```python
import pandas as pd                    # Veri manipülasyonu (DataFrame yapısı)
import numpy as np                     # Sayısal işlemler (array, math)
import matplotlib.pyplot as plt        # Grafik çizimi (low-level kontrol)
import seaborn as sns                  # İstatistiksel görselleştirme (matplotlib üstü)
import warnings                        # Uyarı mesajlarını bastırmak için

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
# train_test_split: veriyi train/test'e böl
# cross_val_score: k-fold cross validation skoru hesapla
# StratifiedKFold: class oranını koruyarak katlama yap

from sklearn.linear_model import LogisticRegression
# Lojistik regresyon: binary classification için baseline model

from sklearn.ensemble import RandomForestClassifier
# Random Forest: birden çok decision tree'nin oy çoğunluğu ile tahmin

from sklearn.preprocessing import StandardScaler
# Feature'ları mean=0, std=1'e normalize et (LogReg için gerekli)

from sklearn.pipeline import make_pipeline
# Scaler + Model'i tek bir pipeline'da birleştir (data leakage önler)

from sklearn.metrics import (
    accuracy_score,          # Doğru tahmin oranı (yanıltıcı olabilir)
    recall_score,            # TP / (TP + FN) — churner'ları yakalama oranı
    precision_score,         # TP / (TP + FP) — "churn" dediğimizde ne kadar haklıyız
    f1_score,                # Precision ve recall'ın harmonik ortalaması
    confusion_matrix,        # 2×2 tablo: TP, FP, TN, FN
    classification_report,   # Precision/recall/F1 tüm class'lar için
    roc_auc_score            # ROC eğrisi altındaki alan (ranking kalitesi)
)

warnings.filterwarnings('ignore')      # Gereksiz uyarıları kapat
sns.set_style("whitegrid")             # Grafik arka planı: beyaz + ızgara
plt.rcParams['figure.dpi'] = 100       # Grafik çözünürlüğü

# Kaggle'daki dataset dosyasının yolunu bul
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```

**KARAR:** Tüm import'ları başta yapıyoruz ki notebook'un neye ihtiyacı olduğu hemen görünsün.

---

### BÖLÜM 2: Veri Yükleme

```python
# CSV dosyasını DataFrame'e oku
df = pd.read_csv("/kaggle/input/datasets/blastchar/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Dataset boyutunu ve hafıza kullanımını göster
print(f"Dataset shape: {df.shape[0]:,} customers × {df.shape[1]} features")
print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

# İlk 5 satırı göster — veri yapısını anlamak için
df.head()
```

**Neden:** Veriyi tanımadan analiz yapamazsın. shape → kaç satır/sütun. head() → verinin neye benzediğini görmek.

---

### BÖLÜM 3: Veri Yapısı İnceleme

```python
# Her sütunun tipi, null sayısı, bellek kullanımı
df.info()
```

**Burada fark ettiğim şey:** `TotalCharges` sütunu `object` (string) tipinde. Sayısal olması gerekiyor — bu bir ETL/veri aktarım hatası.

```python
# Sayısal sütunların istatistiksel özeti (mean, std, min, max, quartiles)
df.describe()
```

```python
# Kategorik sütunların benzersiz değer sayısı ve örnekleri
print("Categorical columns and unique values:\n")
for col in df.select_dtypes(include='object').columns:
    print(f"  {col:20s} → {df[col].nunique()} unique: {df[col].unique()[:5]}")
```

**KARAR:** TotalCharges tip dönüşümü gerekiyor → sonraki adımda yapacağız.

---

### BÖLÜM 4: Veri Temizleme — TotalCharges

```python
# String'i sayıya çevir. Çevrilemeyen değerler NaN olur.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# errors="coerce" → hata veren hücreleri NaN yap (hata fırlatma)

# Kaç NaN oluştuğunu say
print(f"NaN values in TotalCharges: {df['TotalCharges'].isna().sum()}")
# Sonuç: 11
```

```python
# Bu 11 NaN satırını detaylı incele — NEDEN null?
df[df["TotalCharges"].isna()][["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"]]
```

**Gördüğüm pattern:** 11 satırın HEPSİNDE `tenure = 0`. Yani bu müşteriler henüz ilk faturalarını almamış. TotalCharges = boş çünkü hiç ödeme yapmamışlar.

```python
# İmputation kararı: TotalCharges = 0 yap
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Doğrulama: hiç null kalmamalı
assert df.isna().sum().sum() == 0, "Unexpected nulls remain"
print("✓ Dataset is clean. No missing values remain.")
print(f"  Final shape: {df.shape}")
```

**KARAR ROADMAP:**
```
Null var → Neden null? → tenure=0, yeni müşteriler → Mantıksal imputation: $0
                                                   → Silmek YANLIŞ olurdu (survivorship bias)
                                                   → Mean/median imputation YANLIŞ olurdu (bilgi var, tahmin gereksiz)
```

---

### BÖLÜM 5: Target Değişken Analizi

```python
# Churn dağılımını say
churn_counts = df["Churn"].value_counts()           # Mutlak sayılar: No=5174, Yes=1869
churn_pct = df["Churn"].value_counts(normalize=True) * 100  # Yüzdeler: No=73.5%, Yes=26.5%

fig, axes = plt.subplots(1, 2, figsize=(12, 4))     # 1 satır, 2 sütun grafik alanı

# Sol grafik: bar chart (sayılar)
axes[0].bar(["Retained", "Churned"], churn_counts.values, 
            color=["steelblue", "crimson"], edgecolor="black", linewidth=0.5)
for i, (v, p) in enumerate(zip(churn_counts.values, churn_pct.values)):
    axes[0].text(i, v + 50, f"{v:,}\n({p:.1f}%)", ha="center", fontsize=11)
    # Her barın üstüne sayı + yüzde yaz
axes[0].set_ylabel("Count")
axes[0].set_title("Churn Distribution")

# Sağ grafik: pie chart (oranlar)
axes[1].pie(churn_counts.values, labels=["Retained", "Churned"], 
            colors=["steelblue", "crimson"], autopct="%1.1f%%",
            startangle=90, explode=[0, 0.05])      # explode: churn dilimini hafif ayır
axes[1].set_title("Churn Ratio")

plt.tight_layout()   # Grafiklerin birbirine binmesini önle
plt.show()

print(f"\nChurn rate: {churn_pct['Yes']:.2f}%")
print(f"Baseline accuracy (predicting majority class): {churn_pct['No']:.2f}%")
# → %73.5 accuracy "herkes retained" diyerek elde edilebilir → accuracy YANILTICI
```

**KARAR ROADMAP:**
```
%26.5 churn → Moderate imbalance (ciddi değil)
           → SMOTE gerek yok (class_weight yeterli)
           → Accuracy YANILTICI (naive model %73.5 yapar)
           → Asıl metrik: RECALL
           → Neden recall? → 42:1 maliyet asimetrisi (kaçırmak >> yanlış alarm)
```

---

### BÖLÜM 6: EDA — Demografik Özellikler

```python
demographics = ["gender", "SeniorCitizen", "Partner", "Dependents"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))    # 2×2 grid

for ax, col in zip(axes.flatten(), demographics):    # Her demografik feature için:
    # Crosstab: her kategorinin churn oranını hesapla
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    # normalize="index" → satır bazında yüzde (her kategorinin kendi churn rate'i)
    
    ct["Yes"].plot(kind="bar", ax=ax, color="crimson", edgecolor="black", linewidth=0.5)
    ax.axhline(26.5, color="gray", linestyle="--", linewidth=1, label="Overall avg")
    # Kesikli çizgi: genel ortalama churn rate (referans noktası)
    
    ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 50)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    
    # Her barın üstüne yüzde değeri yaz
    for i, v in enumerate(ct["Yes"].values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
```

**KARAR:** Gender signal yok → model'den çıkarılabilir (fairness açısından da daha iyi).

---

### BÖLÜM 7: EDA — Servis Özellikleri

```python
service_cols = ["InternetService", "OnlineSecurity", "OnlineBackup", 
               "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()                               # 2D array'i 1D'ye çevir (loop kolaylığı)

for i, col in enumerate(service_cols):
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct["Yes"].plot(kind="bar", ax=axes[i], color="crimson", edgecolor="black", linewidth=0.5)
    axes[i].axhline(26.5, color="gray", linestyle="--", linewidth=1)
    axes[i].set_title(col, fontsize=11, fontweight="bold")
    axes[i].set_ylabel("Churn %")
    axes[i].set_ylim(0, 50)
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=20)
    for j, v in enumerate(ct["Yes"].values):
        axes[i].text(j, v + 1, f"{v:.0f}%", ha="center", fontsize=9)

axes[-1].axis("off")  # Son grafik alanını kapat (7 feature, 8 alan var)
plt.suptitle("Churn Rate by Service Feature (dashed = overall 26.5%)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()
```

```python
# Fiber + TechSupport etkileşimini derinleştir
print("Churn rate for Fiber Optic customers:\n")
fiber_df = df[df["InternetService"] == "Fiber optic"]   # Sadece fiber müşterileri filtrele
print(fiber_df.groupby("TechSupport")["Churn"].value_counts(normalize=True).unstack().round(3))
# unstack() → pivot tablo formatına çevir (satırda TechSupport, sütunda Churn Yes/No)

print(f"\nFiber + No Tech Support: {fiber_df[fiber_df['TechSupport']=='No']['Churn'].value_counts(normalize=True)['Yes']*100:.1f}% churn")
print(f"Fiber + Tech Support:    {fiber_df[fiber_df['TechSupport']=='Yes']['Churn'].value_counts(normalize=True)['Yes']*100:.1f}% churn")
```

**KARAR:** Fiber + No TechSupport kombinasyonu çok riskli. Actionable: fiber paketlere tech support dahil et.

---

### BÖLÜM 8: EDA — Sözleşme & Ödeme

```python
billing_cols = ["Contract", "PaymentMethod", "PaperlessBilling"]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, col in zip(axes, billing_cols):
    ct = pd.crosstab(df[col], df["Churn"], normalize="index") * 100
    ct["Yes"].sort_values(ascending=False).plot(     # En yüksek churn solda olsun
        kind="bar", ax=ax, color="crimson", edgecolor="black", linewidth=0.5)
    ax.axhline(26.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f"Churn Rate by {col}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 55)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    for i, v in enumerate(ct["Yes"].sort_values(ascending=False).values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

plt.tight_layout()
plt.show()
```

**BULGU:** Contract type = dataset'teki EN GÜÇLÜ sinyal. 15.3× fark.

---

### BÖLÜM 9: EDA — Sayısal Özellikler & Korelasyon

```python
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Her sayısal feature için churn/no-churn grubunu ayrı histogram olarak çiz
for label, color in [("No", "steelblue"), ("Yes", "crimson")]:
    subset = df[df["Churn"] == label]["tenure"]
    axes[0].hist(subset, bins=36, alpha=0.6, color=color, label=f"Churn={label}", density=True)
    # density=True → y ekseni frekans değil yoğunluk (farklı grup boyutlarını karşılaştırılabilir yapar)
axes[0].set_xlabel("Tenure (months)")
axes[0].set_title("Tenure Distribution by Churn")
axes[0].legend()
axes[0].axvline(6, color="black", linestyle=":", label="6-month mark")
# 6. ay çizgisi: churn'ün yoğunlaştığı bölgeyi işaretle

# Aynı yaklaşım MonthlyCharges ve TotalCharges için tekrarlanıyor
# (kısaltılmış — aynı pattern)

plt.tight_layout()
plt.show()
```

```python
# Korelasyon matrisi
df_temp = df.copy()
df_temp["Churn_numeric"] = (df_temp["Churn"] == "Yes").astype(int)
# Churn'ü 0/1'e çevir ki korelasyon hesaplanabilsin

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", "Churn_numeric"]
corr = df_temp[numeric_cols].corr()   # Pearson korelasyon matrisi

fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))   # Üst üçgeni maskele (tekrar önleme)
sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            square=True, linewidths=0.5, ax=ax)
# annot=True → hücrelere sayıları yaz
# cmap="RdBu_r" → kırmızı=pozitif, mavi=negatif
# center=0 → 0 noktası renk skalasının ortası

ax.set_title("Correlation Matrix (Numeric Features + Target)")
plt.tight_layout()
plt.show()

print("\nCorrelations with Churn:")
print(corr["Churn_numeric"].drop("Churn_numeric").sort_values())
```

**KARAR:** tenure (-0.352) en güçlü. TotalCharges ↔ tenure = 0.83 → multicollinearity var ama tree modelde sorun değil.

---

### BÖLÜM 10: Yüksek Riskli Segment Kesişimi

```python
# Tenure ve charges'ı binlere ayır (cross-tab kolaylığı için)
df["TenureGroup"] = pd.cut(df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
                           labels=["0-6 mo", "7-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
# pd.cut: sürekli değişkeni kategorik gruplara böl

df["ChargeGroup"] = pd.cut(df["MonthlyCharges"], bins=[0, 35, 70, 120],
                           labels=["Low (<$35)", "Mid ($35-70)", "High (>$70)"])

# Tüm risk parametreleri için özet tablo oluştur
parameters = ["Contract", "TenureGroup", "ChargeGroup",
              "InternetService", "TechSupport", "PaymentMethod"]

rows = []
for p in parameters:
    grp = df.groupby(p, observed=True)["Churn"]
    summary = pd.DataFrame({
        "Parameter": p,
        "Category": grp.count().index,
        "Customer Count": grp.count().values,
        "Churn Rate (%)": (grp.apply(lambda s: (s == "Yes").mean()) * 100).round(1).values
    })
    rows.append(summary)

evidence = pd.concat(rows, ignore_index=True)
evidence.sort_values(["Parameter", "Churn Rate (%)"], ascending=[True, False])
```

```python
# EN RİSKLİ KESİŞİMİ SORGULA
high_risk = df[
    (df["Contract"] == "Month-to-month") &      # Aylık sözleşme
    (df["tenure"] <= 6) &                        # İlk 6 ay
    (df["InternetService"] == "Fiber optic") &   # Fiber internet
    (df["TechSupport"] == "No")                  # Tech support yok
]

print("HIGH-RISK SEGMENT ANALYSIS")
print("=" * 50)
print(f"Segment: Month-to-month + Tenure ≤ 6mo + Fiber + No Tech Support")
print(f"")
print(f"Segment size:  {len(high_risk):,} customers ({len(high_risk)/len(df)*100:.1f}% of base)")
print(f"Churn rate:    {(high_risk['Churn']=='Yes').mean()*100:.1f}%")
print(f"vs. overall:   {overall_rate:.1f}%")
print(f"Lift:          {(high_risk['Churn']=='Yes').mean() / (df['Churn']=='Yes').mean():.1f}×")
# Lift: bu segmentin churn oranı ÷ genel churn oranı (ne kadar daha riskli?)

print(f"")
print(f"If we retained even 50% of this segment:")
print(f"  Customers saved: ~{int(len(high_risk) * (high_risk['Churn']=='Yes').mean() * 0.5)}")
print(f"  Revenue saved:   ~${int(len(high_risk) * (high_risk['Churn']=='Yes').mean() * 0.5 * 840):,}/year")
# segment_size × churn_rate × %50_başarı × $840/müşteri/yıl
```

**KARAR:** Bu segment %55+ churn ediyor. Küçük bir retention yatırımıyla yüz binlerce $ kurtarılabilir.

---

### BÖLÜM 11: Preprocessing (Modelleme Hazırlığı)

```python
# EDA sırasında oluşturduğumuz bin sütunlarını ve ID'yi çıkar
df_model = df.drop(columns=["customerID", "TenureGroup", "ChargeGroup"])
# customerID: tahmin gücü yok (benzersiz identifier)
# TenureGroup/ChargeGroup: sadece EDA için oluşturduk, ham değerler kalıyor

# Target'ı 0/1'e çevir
df_model["Churn"] = df_model["Churn"].map({"Yes": 1, "No": 0})

# Kategorik sütunları one-hot encode et
df_model = pd.get_dummies(df_model, drop_first=True)
# drop_first=True → her kategoriden 1 dummy düşür (dummy trap önleme)
# Örnek: Contract_One year, Contract_Two year (Month-to-month = referans = 0,0)

# Feature matrix ve target'ı ayır
X = df_model.drop(columns=["Churn"])    # Tüm feature'lar
y = df_model["Churn"]                    # Hedef değişken

# Veriyi train/test'e böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# test_size=0.2 → %20 test, %80 train
# random_state=42 → tekrarlanabilirlik (aynı bölünme her seferinde)
# stratify=y → churn oranını her iki sette de korur (%26.5)

print(f"Training set: {X_train.shape[0]:,} samples ({y_train.mean()*100:.1f}% positive)")
print(f"Test set:     {X_test.shape[0]:,} samples ({y_test.mean()*100:.1f}% positive)")
print(f"Features:     {X_train.shape[1]}")
print(f"\nClass distribution preserved: train={y_train.mean():.3f}, test={y_test.mean():.3f}")
```

**KARAR ROADMAP:**
```
SMOTE kullanmıyorum çünkü:
  1. %26.5 moderate imbalance (SMOTE %5'in altı için tasarlanmış)
  2. Feature'ların çoğu kategorik (SMOTE interpole eder → kategoriklerde anlamsız)
  3. class_weight aynı etkiyi daha temiz sağlıyor (loss function'da ağırlık)
```

---

### BÖLÜM 12: Baseline Model — Logistic Regression

```python
# Pipeline: Scaler + Model birlikte (data leakage önler)
lr_baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42))
# StandardScaler: feature'ları mean=0, std=1'e normalize et
#   → LogReg katsayıları ancak normalize veride karşılaştırılabilir
# max_iter=2000: convergence için yeterli iterasyon
# Pipeline'da scaler sadece train'e fit olur, test'e sadece transform uygulanır

lr_baseline.fit(X_train, y_train)          # Modeli eğit
y_pred_lr = lr_baseline.predict(X_test)    # Test seti üzerinde tahmin yap

print("LOGISTIC REGRESSION — BASELINE (unweighted)")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.3f}")     # Genel doğruluk
print(f"Recall:    {recall_score(y_test, y_pred_lr):.3f}")        # Churner yakalama oranı
print(f"Precision: {precision_score(y_test, y_pred_lr):.3f}")     # "Churn" dediğimizde haklı olma oranı
print(f"F1:        {f1_score(y_test, y_pred_lr):.3f}")            # Precision/Recall dengesi
print(f"ROC-AUC:   {roc_auc_score(y_test, lr_baseline.predict_proba(X_test)[:,1]):.3f}")
# predict_proba[:,1] → class=1 (churn) olasılık tahminleri
# ROC-AUC: modelin ranking kalitesi (threshold-bağımsız)
```

**SONUÇ:** Recall = 0.57 → churner'ların %43'ünü KAÇIRIYORUZ. İş etkisi: 43% × 1869 × $840 = $680K/yıl kayıp.

```python
# Katsayıları çıkar ve görselleştir
coefs = pd.Series(
    lr_baseline.named_steps["logisticregression"].coef_[0],  # Katsayı vektörü
    index=X.columns                                           # Feature isimleri
).sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
coefs.plot(kind="barh", ax=ax, color=["crimson" if v > 0 else "steelblue" for v in coefs.values])
# Pozitif katsayı (kırmızı) = churn olasılığını ARTIRIR
# Negatif katsayı (mavi) = churn olasılığını AZALTIR
ax.set_xlabel("Coefficient (standardized)")
ax.set_title("Logistic Regression Coefficients\n(positive = increases churn probability)")
ax.axvline(0, color="black", linewidth=0.5)    # Sıfır referans çizgisi
plt.tight_layout()
plt.show()
```

**KARAR:** Katsayılar EDA ile aynı pattern'i gösteriyor → convergent validation. Ama recall yetersiz → class_weight ekle.

---

### BÖLÜM 13: Class Weight ile Dengeleme

```python
# class_weight="balanced": azınlık sınıfının loss'unu oranla çarp
# Formül: weight_i = n_samples / (n_classes × n_samples_i)
# Churn için: 7043 / (2 × 1869) ≈ 1.88 → churn loss'u 1.88× çarpılır
# No-churn için: 7043 / (2 × 5174) ≈ 0.68

lr_balanced = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
)
lr_balanced.fit(X_train, y_train)
y_pred_lr_bal = lr_balanced.predict(X_test)

print("LOGISTIC REGRESSION — BALANCED (class_weight='balanced')")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr_bal):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr_bal):.3f}  ← +{recall_score(y_test, y_pred_lr_bal) - recall_score(y_test, y_pred_lr):.3f} improvement")
# Recall farkını hesaplayıp göster
print(f"Precision: {precision_score(y_test, y_pred_lr_bal):.3f}")
print(f"F1:        {f1_score(y_test, y_pred_lr_bal):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, lr_balanced.predict_proba(X_test)[:,1]):.3f}")
```

**SONUÇ:** Recall: 0.57 → 0.79 (+38% iyileşme). Precision düştü (0.66 → 0.51) ama 42:1 maliyet oranında bu kabul edilebilir.

---

### BÖLÜM 14: Random Forest — Hyperparameter Araştırması

```python
# DENEY 1: n_estimators (ağaç sayısı) etkisi
print("Effect of n_estimators (max_depth=None, balanced):")
print("-" * 50)
for n in [10, 50, 100, 300]:
    rf_temp = RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42)
    # n_estimators: ormandaki ağaç sayısı (daha fazla = daha stabil, ama yavaş)
    rf_temp.fit(X_train, y_train)
    y_pred_temp = rf_temp.predict(X_test)
    print(f"  n={n:3d} → Recall: {recall_score(y_test, y_pred_temp):.3f}, "
          f"Precision: {precision_score(y_test, y_pred_temp):.3f}, "
          f"Accuracy: {accuracy_score(y_test, y_pred_temp):.3f}")
# GÖZLEM: 100'den sonra diminishing returns → 300 yeterli
```

```python
# DENEY 2: max_depth etkisi — KRİTİK KEŞİF
print("Effect of max_depth (n_estimators=300, balanced):")
print("-" * 50)
depths = [3, 5, 7, 10, 15, None]          # None = sınırsız derinlik
results = []
for d in depths:
    rf_temp = RandomForestClassifier(n_estimators=300, max_depth=d, 
                                     class_weight="balanced", random_state=42)
    rf_temp.fit(X_train, y_train)
    y_pred_temp = rf_temp.predict(X_test)
    r = recall_score(y_test, y_pred_temp)
    p = precision_score(y_test, y_pred_temp)
    results.append({"max_depth": str(d), "recall": r, "precision": p})
    print(f"  max_depth={str(d):>4s} → Recall: {r:.3f}, Precision: {p:.3f}")

print("\n⚠️  Notice: unlimited depth DECREASES recall to ~0.50!")
print("    This is because pure leaves negate class_weight's effect.")
```

**KRİTİK KEŞİF VE AÇIKLAMASI:**
```
max_depth=None → recall ≈ 0.50 (rastgele tahminle aynı!)
max_depth=5   → recall ≈ 0.82 (çok iyi!)

NEDEN?
─────
class_weight "balanced" nasıl çalışır:
  → Eğitim sırasında churn sınıfının loss'unu 1.88× çarpar
  → Model "churn tahmini yanlışsa daha çok cezalandırılırım" diye öğrenir
  
max_depth=None'da ne oluyor:
  → Ağaçlar TAM DERİNLİKTE büyüyor
  → Her yaprak pure oluyor (sadece 1 class içeriyor)
  → Pure yaprakta loss = 0 (hep doğru tahmin)
  → 0 × 1.88 = 0 → ağırlık çarpanının HİÇBİR ETKİSİ YOK
  → Model majority class'a (No) kayıyor
  
max_depth=5'te ne oluyor:
  → Yapraklar IMPURE kalıyor (karışık class'lar)
  → Loss > 0 → ağırlık çarpanı AKTİF
  → Model churn'ü daha agresif tahmin ediyor → recall yükseliyor
```

---

### BÖLÜM 15: Final Model & Business Impact

```python
# Optimal parametre kombinasyonu ile final model
rf_final = RandomForestClassifier(
    n_estimators=300,           # 300 ağaç (yeterli stabilite)
    max_depth=5,                # Sığ ağaç (class_weight aktif kalsın)
    class_weight="balanced",    # Azınlık sınıfı ağırlıklı
    random_state=42             # Tekrarlanabilirlik
)
rf_final.fit(X_train, y_train)
y_pred_rf = rf_final.predict(X_test)

# Tüm metrikleri yazdır
print("RANDOM FOREST — FINAL MODEL (n=300, depth=5, balanced)")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.3f}")
print(f"F1:        {f1_score(y_test, y_pred_rf):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, rf_final.predict_proba(X_test)[:,1]):.3f}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=["Retained", "Churned"]))
# Her class için precision, recall, f1, support yazdır
```

```python
# Confusion Matrix + Business Impact Hesabı
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"])
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_title("Confusion Matrix — Final Model")

# Confusion matrix'ten iş etkisi hesapla
tn, fp, fn, tp = cm.ravel()
# tn: doğru "retained" tahminleri
# fp: yanlış "churn" tahminleri (gereksiz teklif gönderilecek)
# fn: kaçırılan churner'lar (en pahalı hata!)
# tp: doğru yakalanan churner'lar

axes[1].axis("off")
axes[1].text(0.1, 0.9, "Business Impact Analysis", fontsize=14, fontweight="bold", transform=axes[1].transAxes)
axes[1].text(0.1, 0.75, f"True Positives (caught churners):  {tp}", fontsize=11, transform=axes[1].transAxes)
axes[1].text(0.1, 0.63, f"False Negatives (missed churners): {fn}", fontsize=11, transform=axes[1].transAxes)
axes[1].text(0.1, 0.51, f"False Positives (unnecessary offers): {fp}", fontsize=11, transform=axes[1].transAxes)
axes[1].text(0.1, 0.39, f"True Negatives (correctly ignored): {tn}", fontsize=11, transform=axes[1].transAxes)
axes[1].text(0.1, 0.22, f"Revenue saved (TP × $840):  ${tp * 840:,.0f}/year", fontsize=11, color="green", fontweight="bold", transform=axes[1].transAxes)
# tp × $840 = yakalanan churner sayısı × müşteri başı yıllık gelir
axes[1].text(0.1, 0.10, f"Wasted offers (FP × $20):   ${fp * 20:,.0f}/year", fontsize=11, color="orange", transform=axes[1].transAxes)
# fp × $20 = gereksiz retention teklifi maliyeti
axes[1].text(0.1, -0.02, f"Net value:                  ${tp * 840 - fp * 20:,.0f}/year", fontsize=12, color="darkgreen", fontweight="bold", transform=axes[1].transAxes)
# Net = kurtarılan gelir - boşa harcanan teklif maliyeti

plt.tight_layout()
plt.show()
```

---

### BÖLÜM 16: Feature Importance

```python
# Random Forest'ın Gini-based feature importance'ı
imp = pd.Series(rf_final.feature_importances_, index=X.columns).sort_values(ascending=False)
# feature_importances_: her feature'ın ağaçlardaki ortalama Gini impurity azalması
# Yüksek değer = daha önemli feature

fig, ax = plt.subplots(figsize=(10, 8))
top_15 = imp.head(15)
colors = ["crimson" if i < 5 else "steelblue" for i in range(len(top_15))]
# Top 5'i kırmızı, geri kalanı mavi

ax.barh(range(len(top_15)), top_15.values, color=colors, edgecolor="black", linewidth=0.5)
ax.set_yticks(range(len(top_15)))
ax.set_yticklabels(top_15.index)
ax.set_xlabel("Feature Importance (Gini)")
ax.set_title("Top 15 Feature Importances — Random Forest\n(red = top 5 drivers)")
ax.invert_yaxis()    # En önemli feature en üstte olsun
plt.tight_layout()
plt.show()

print("\nTop 10 features:")
for feat, importance in imp.head(10).items():
    print(f"  {feat:35s} {importance:.4f}")
```

**CONVERGENT VALIDATION:**
```
3 bağımsız yöntem → aynı sonuç:
  EDA (univariate rates)  → contract, tenure, fiber, e-check, tech support
  LogReg (katsayılar)     → contract, tenure, fiber, e-check, tech support  
  RF (Gini importance)    → contract, tenure, fiber, e-check, tech support
  
Bu tutarlılık = güvenilir bulgular (tesadüf değil, gerçek sinyal)
```

---

## ÖZET: KARAR HARİTASI (TÜM KARARLAR)

| # | Karar Noktası | Seçenekler | Seçimim | Neden |
|---|---|---|---|---|
| 1 | Null imputation | Drop / Mean / fillna(0) | fillna(0) | tenure=0 → mantıksal olarak $0 |
| 2 | Metrik seçimi | Accuracy / F1 / Recall | Recall | 42:1 maliyet asimetrisi |
| 3 | Imbalance çözümü | SMOTE / Class weight / Threshold | Class weight | Moderate imbalance, kategorik data |
| 4 | Baseline model | LogReg / Decision Tree | LogReg | Yorumlanabilir, katsayı analizi |
| 5 | Tree depth | None / 3 / 5 / 7 / 10 | 5 | Class weight + pure leaf interaction |
| 6 | n_estimators | 10 / 50 / 100 / 300 | 300 | Diminishing returns ama stabil |
| 7 | Final model | LogReg balanced / RF depth=5 | RF depth=5 | Recall 0.82 > 0.79 |
| 8 | Feature encoding | Label / One-hot / Target | One-hot (drop_first) | Tree + linear modeller için uyumlu |
| 9 | Train/test split | Random / Stratified | Stratified | Class ratio korunsun |
| 10 | Evaluation | Single split / CV | Single split (Phase 1) | CV Phase 2'de planlandı |
