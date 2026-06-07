# =====================================================
# UAS INFORMATION RETRIEVAL
# SISTEM PENCARIAN DOKUMEN ULASAN PRODUK
# DATASET PRDECT-ID
# TF-IDF + MULTINOMIAL NAIVE BAYES
# =====================================================

# =====================================================
# IMPORT LIBRARY
# =====================================================

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# STEMMER
# =====================================================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# =====================================================
# STOPWORD
# =====================================================

stopwords = [
    "dan", "di", "ke", "dari", "yang",
    "untuk", "dengan", "atau", "pada",
    "adalah", "ini", "itu", "dalam",
    "secara", "oleh", "karena", "agar",
    "sebagai", "bahwa", "juga", "lebih",
    "dapat", "menjadi", "antara", "setiap",
    "suatu", "akan", "telah", "sudah",
    "para", "mereka", "kami", "kita",
    "anda", "ia", "saya", "aku"
]
stopwords.extend([
    "yg", "nya", "ga", "gak", "aja",
    "sih", "nih", "kok"
])

# =====================================================
# JUMLAH DATA YANG DITAMPILKAN
# =====================================================

DETAIL_DATA = 5

# =====================================================
# MEMBACA DATASET
# =====================================================

print("\n================================================")
print("MEMBACA DATASET")
print("================================================")

dataset_path = "PRDECT-ID Dataset.csv"

df = pd.read_csv(dataset_path)

print(df.head())

# =====================================================
# INFORMASI DATASET
# =====================================================

print("\n================================================")
print("INFORMASI DATASET")
print("================================================")

print(df.info())
print("\nJumlah Data :", len(df))
print("\nDistribusi Emosi :")
print(df["Emotion"].value_counts())

# =====================================================
# MISSING VALUE
# =====================================================

print("\n================================================")
print("MISSING VALUE")
print("================================================")

print(df.isnull().sum())

df = df.dropna()
df = df.reset_index(drop=True)

# =====================================================
# FITUR DAN LABEL
# =====================================================

documents = df["Customer Review"].astype(str)
y = df["Emotion"]

# =====================================================
# PREPROCESSING
# =====================================================

print("\n================================================")
print("DETAIL PREPROCESSING")
print("================================================")

clean_documents = []

for i, text in enumerate(documents):

    text = str(text)

    # CASE FOLDING
    case_folding = text.lower()

    # REMOVE SYMBOL
    remove_symbol = re.sub(r'[^a-zA-Z\s]', ' ', case_folding)

    # TOKENIZING
    tokens = remove_symbol.split()

    # STOPWORD REMOVAL
    filtered = [word for word in tokens if word not in stopwords]

    # STEMMING
    stemmed = [stemmer.stem(word) for word in filtered]

    final_text = " ".join(stemmed)
    clean_documents.append(final_text)

    if i < DETAIL_DATA:

        print("\n====================================")
        print(f"DATA KE-{i+1}")
        print("====================================")

        print("\nDOKUMEN ASLI :")
        print(text)

        print("\nCASE FOLDING :")
        print(case_folding)

        print("\nREMOVE SYMBOL :")
        print(remove_symbol)

        print("\nTOKENIZING :")
        print(tokens)
        print("Jumlah Token :", len(tokens))

        print("\nSTOPWORD REMOVAL :")
        print(filtered)
        print("Jumlah Setelah Stopword :", len(filtered))

        print("\nSTEMMING :")
        print(stemmed)

        print("\nFINAL TEXT :")
        print(final_text)

    if i % 500 == 0:
        print(f"Processed {i} data...")

df["clean_text"] = clean_documents

print("\nPreprocessing selesai")

print("\n================================================")
print("HASIL PREPROCESSING")
print("================================================")

for i in range(min(DETAIL_DATA, len(df))):
    print("\n--------------------------------")
    print("REVIEW ASLI :")
    print(df.iloc[i]["Customer Review"])
    print("\nREVIEW BERSIH :")
    print(df.iloc[i]["clean_text"])

# =====================================================
# SPLIT DATA
# Split SEBELUM TF-IDF untuk menghindari data leakage
# =====================================================

print("\n================================================")
print("SPLIT DATA")
print("================================================")

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["clean_text"],
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training :", len(X_train_raw))
print("Testing  :", len(X_test_raw))

print("\nDistribusi Data Training")
print(y_train.value_counts())

print("\nDistribusi Data Testing")
print(y_test.value_counts())

# =====================================================
# TF-IDF VECTORIZER
# fit_transform hanya pada training, transform pada testing
# =====================================================

print("\n================================================")
print("TF-IDF VECTORIZER")
print("================================================")

vectorizer = TfidfVectorizer(max_features=500)

X_train = vectorizer.fit_transform(X_train_raw)
X_test  = vectorizer.transform(X_test_raw)

fitur = vectorizer.get_feature_names_out()

print("Jumlah Vocabulary :", len(fitur))
print("\n50 Vocabulary Pertama :")
print(fitur[:50])

print("\n================================================")
print("UKURAN MATRIX TF-IDF")
print("================================================")

print("Jumlah Dokumen Training :", X_train.shape[0])
print("Jumlah Dokumen Testing  :", X_test.shape[0])
print("Jumlah Fitur            :", X_train.shape[1])

print("\n================================================")
print("CONTOH BOBOT TF-IDF (5 baris, 20 kolom pertama)")
print("================================================")

tfidf_df = pd.DataFrame(X_train.toarray(), columns=fitur)
print(tfidf_df.iloc[:5, :20])

print("\n================================================")
print("UKURAN DATA TRAINING DAN TESTING")
print("================================================")

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

# =====================================================
# TRAINING MULTINOMIAL NAIVE BAYES
# =====================================================

print("\n================================================")
print("TRAINING MULTINOMIAL NAIVE BAYES")
print("================================================")

model = MultinomialNB()
model.fit(X_train, y_train)

print("Training berhasil")
print("Jumlah kelas :", len(model.classes_))
print("Daftar kelas :")
print(model.classes_)

# =====================================================
# KATA PALING DOMINAN SETIAP EMOSI
# =====================================================

print("\n================================================")
print("KATA PALING DOMINAN SETIAP EMOSI")
print("================================================")

feature_names = vectorizer.get_feature_names_out()

for i, kelas in enumerate(model.classes_):
    top10 = np.argsort(model.feature_log_prob_[i])[-10:]
    print("\nEmosi :", kelas)
    for idx in reversed(top10):
        print(" ", feature_names[idx])

# =====================================================
# PRIOR PROBABILITY
# =====================================================

print("\n================================================")
print("PRIOR PROBABILITY")
print("================================================")

for kelas, prob in zip(model.classes_, np.exp(model.class_log_prior_)):
    print(f"{kelas} : {prob:.4f}")

# =====================================================
# EVALUASI MODEL
# =====================================================

print("\n================================================")
print("EVALUASI MODEL")
print("================================================")

train_pred = model.predict(X_train)
test_pred  = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc  = accuracy_score(y_test, test_pred)

print("Training Accuracy :", round(train_acc, 4))
print("Testing Accuracy  :", round(test_acc, 4))

plt.figure(figsize=(6, 4))
bars = plt.bar(
    ["Train", "Test"],
    [train_acc, test_acc],
    color=["steelblue", "tomato"]
)
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.01,
        f"{round(yval, 4)}",
        ha="center"
    )
plt.title("Perbandingan Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

report = classification_report(y_test, test_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

plt.figure(figsize=(8, 5))
sns.heatmap(
    report_df.iloc[:-3, :-1],
    annot=True,
    cmap="YlGnBu",
    fmt=".2f"
)
plt.title("Classification Report Heatmap")
plt.tight_layout()
plt.show()

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\n================================================")
print("CONFUSION MATRIX")
print("================================================")

cm = confusion_matrix(y_test, test_pred)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
print(cm_df)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =====================================================
# DISTRIBUSI EMOSI
# =====================================================

plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df, x="Emotion")
for p in ax.patches:
    ax.annotate(
        str(p.get_height()),
        (p.get_x() + 0.3, p.get_height() + 10)
    )
plt.title("Distribusi Emosi")
plt.tight_layout()
plt.show()

# =====================================================
# SISTEM PENCARIAN DOKUMEN BERBASIS NAIVE BAYES
#
# Alur:
#   Query → Preprocessing → TF-IDF → Naive Bayes
#   → Prediksi Emosi → Retrieve Dokumen → Tampilkan
#
# Naive Bayes berperan sebagai mesin retrieval
# berbasis kategori (category-based retrieval).
# Query diklasifikasikan ke kelas emosi tertentu,
# lalu sistem mengembalikan seluruh dokumen
# yang berada pada kelas emosi yang sama.
# =====================================================

print("\n================================================")
print("SISTEM PENCARIAN DOKUMEN BERBASIS NAIVE BAYES")
print("================================================")

query = input("\nMasukkan Query Pencarian : ")

# ------ PREPROCESSING QUERY ------

print("\n================================================")
print("PREPROCESSING QUERY")
print("================================================")

print("\nQUERY ASLI :")
print(query)

case_folding  = query.lower()
print("\nCASE FOLDING :")
print(case_folding)

remove_symbol = re.sub(r'[^a-zA-Z\s]', ' ', case_folding)
print("\nREMOVE SYMBOL :")
print(remove_symbol)

tokens   = remove_symbol.split()
print("\nTOKENIZING :")
print(tokens)

filtered = [word for word in tokens if word not in stopwords]
print("\nSTOPWORD REMOVAL :")
print(filtered)

stemmed  = [stemmer.stem(word) for word in filtered]
print("\nSTEMMING :")
print(stemmed)

query_clean = " ".join(stemmed)
print("\nFINAL QUERY :")
print(query_clean)

if query_clean.strip() == "":
    print("\nQuery kosong setelah preprocessing!")
    exit()

# ------ TF-IDF QUERY ------

print("\n================================================")
print("TF-IDF QUERY")
print("================================================")

query_vector = vectorizer.transform([query_clean])
query_array  = query_vector.toarray()[0]

print("\nTop Term Query :")
top_terms = np.argsort(query_array)[::-1][:10]
for idx in top_terms:
    if query_array[idx] > 0:
        print(f"  {fitur[idx]} = {round(query_array[idx], 4)}")

# ------ PREDIKSI EMOSI QUERY ------
#
# Naive Bayes menghitung P(Emosi | Query) untuk
# setiap kelas, lalu memilih kelas tertinggi.
# Inilah inti retrieval berbasis Naive Bayes:
# query dipetakan ke satu kelas emosi sebagai
# dasar pengambilan dokumen.

print("\n================================================")
print("PREDIKSI EMOSI QUERY")
print("================================================")

prediction  = model.predict(query_vector)
probability = model.predict_proba(query_vector)

prediksi_emosi = prediction[0]

print("\nQuery          :", query)
print("Prediksi Emosi :", prediksi_emosi)

print("\nProbabilitas setiap kelas :")
prob_df = pd.DataFrame({
    "Emotion"    : model.classes_,
    "Probability": probability[0]
}).sort_values(by="Probability", ascending=False)

for _, row in prob_df.iterrows():
    print(f"  {row['Emotion']:<10} : {row['Probability']*100:.2f}%")

# Visualisasi probabilitas prediksi emosi query
plt.figure(figsize=(7, 4))
colors = [
    "tomato" if e == prediksi_emosi else "steelblue"
    for e in prob_df["Emotion"]
]
plt.bar(prob_df["Emotion"], prob_df["Probability"], color=colors)
plt.title(
    f"Probabilitas Prediksi Emosi Query\n"
    f"(merah = emosi terpilih: {prediksi_emosi})"
)
plt.xlabel("Emotion")
plt.ylabel("P(Emosi | Query)")
plt.tight_layout()
plt.show()

# ------ RETRIEVAL DOKUMEN ------
#
# Sistem mengambil seluruh dokumen yang emosinya
# sama dengan hasil prediksi Naive Bayes.
# Ini adalah category-based retrieval:
# klasifikasi query menentukan koleksi dokumen
# yang akan dikembalikan sebagai hasil pencarian.

print("\n================================================")
print(f"HASIL PENCARIAN - EMOSI : {prediksi_emosi}")
print("================================================")

hasil = df[df["Emotion"] == prediksi_emosi].reset_index(drop=True)

print(f"\nDitemukan {len(hasil)} dokumen dengan emosi '{prediksi_emosi}'")
print("Menampilkan 10 dokumen teratas :\n")

if len(hasil) == 0:
    print("Tidak ada dokumen ditemukan.")
    exit()

for i in range(min(10, len(hasil))):
    print("--------------------------------")
    print(f"Dokumen ke-{i+1}")
    print(f"Emotion  : {hasil.iloc[i]['Emotion']}")
    print(f"Review   : {hasil.iloc[i]['Customer Review']}")
    print()

# Visualisasi jumlah dokumen per emosi
# dengan highlight pada emosi hasil prediksi
emosi_counts = df["Emotion"].value_counts()
colors_bar = [
    "tomato" if e == prediksi_emosi else "steelblue"
    for e in emosi_counts.index
]

plt.figure(figsize=(8, 4))
plt.bar(emosi_counts.index, emosi_counts.values, color=colors_bar)
plt.title(
    f"Jumlah Dokumen per Emosi\n"
    f"(merah = hasil retrieval: {prediksi_emosi}, "
    f"{len(hasil)} dokumen)"
)
plt.xlabel("Emotion")
plt.ylabel("Jumlah Dokumen")
for i, (emosi, jumlah) in enumerate(emosi_counts.items()):
    plt.text(i, jumlah + 5, str(jumlah), ha="center", fontsize=9)
plt.tight_layout()
plt.show()

print("\n================================================")
print("PROGRAM SELESAI")
print("================================================")