import streamlit as st
import pandas as pd

# --------------------------------------------------
# Konfigurasi halaman (kalau ini file utamanya)
# --------------------------------------------------
st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Load data + simpan ke session_state
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    return df

df = load_data()

if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = df

# Hitung ringkasan dasar
n_rows = df.shape[0]
n_cols = df.shape[1]
feature_cols = [c for c in df.columns if c not in ["id", "stroke"]]
n_features = len(feature_cols)

stroke_counts = df["stroke"].value_counts().sort_index()
n_no_stroke = int(stroke_counts.get(0, 0))
n_stroke = int(stroke_counts.get(1, 0))
prevalence = n_stroke / n_rows * 100 if n_rows > 0 else 0

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <h1 style='margin-bottom:0px;'>🧠 Stroke Prediction Dashboard</h1>
    <p style='color:#666; font-size:16px; margin-top:4px;'>
        Analisis risiko stroke berbasis data klinis dan demografis dari
        <em>Healthcare Stroke Prediction Dataset (Kaggle)</em>.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --------------------------------------------------
# 1) Ringkasan cepat (info cards)
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Pasien", f"{n_rows:,}")
with col2:
    st.metric("Prevalensi Stroke", f"{prevalence:.2f} %", help="Persentase pasien dengan label stroke = 1")
with col3:
    st.metric("Jumlah Fitur Awal", f"{n_features}", help="Tidak termasuk kolom id dan target stroke")

st.caption(
    "Dataset berisi informasi demografis (umur, jenis kelamin, status pernikahan, dll.) "
    "dan faktor klinis (hipertensi, penyakit jantung, kadar gula darah, BMI) yang digunakan "
    "untuk memprediksi apakah seorang pasien pernah mengalami stroke."
)

# --------------------------------------------------
# 2) Latar belakang & tujuan
# --------------------------------------------------
st.markdown("### 🎯 Latar Belakang")

st.markdown(
    """
Stroke merupakan salah satu penyebab kematian dan kecacatan utama secara global.
Deteksi dini individu dengan **risiko stroke tinggi** penting untuk:

- Mengarahkan pasien ke pemeriksaan lanjutan lebih cepat  
- Membantu tenaga kesehatan dalam pengambilan keputusan klinis  
- Mengoptimalkan alokasi sumber daya kesehatan

Dengan memanfaatkan data klinis & demografis, kita membangun **model klasifikasi** yang dapat
mengestimasi risiko stroke pada pasien, dengan fokus utama pada **sensitivitas (recall)** agar
sebanyak mungkin kasus stroke tidak terlewat (*False Negative* serendah mungkin).
"""
)

st.markdown("### 🎯 Tujuan Analisis")

st.markdown(
    """
1. Menganalisis karakteristik data (ketidakseimbangan kelas, missing values, dan outlier).  
2. Melakukan **pre-processing** komprehensif (imputasi, winsorizing, encoding, scaling, dan SMOTE).  
3. Melakukan **feature selection/feature importance** untuk mengidentifikasi faktor paling berpengaruh.  
4. Membangun dan membandingkan beberapa **model klasifikasi** (Logistic Regression, Random Forest, SVM).  
5. Mengevaluasi model menggunakan **training–testing, repeated holdout, dan stratified k-fold cross validation**  
   dengan metrik: akurasi, sensitivitas, spesifisitas, ROC, dan AUC.  
"""
)

# --------------------------------------------------
# 3) Ringkasan dataset
# --------------------------------------------------
st.markdown("### 📁 Ringkasan Dataset")

col4, col5 = st.columns([2, 3])

with col4:
    st.write("**Struktur Kolom (head)**")
    st.dataframe(df.head(), use_container_width=True)

with col5:
    st.write("**Distribusi Target (Stroke)**")
    stroke_df = stroke_counts.reset_index()
    stroke_df.columns = ["stroke", "count"]
    stroke_df["label"] = stroke_df["stroke"].map({0: "Tidak Stroke (0)", 1: "Stroke (1)"})

    # Plotly bar sederhana (biar sama gaya dengan page lain)
    import plotly.express as px

    fig = px.bar(
        stroke_df,
        x="label",
        y="count",
        text="count",
        color="label",
        color_discrete_sequence=px.colors.sequential.Blues,
        labels={"label": "Kelas", "count": "Jumlah Pasien"},
        title="Distribusi Kelas Target"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=350, margin=dict(t=60, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.info(
    f"Dari total **{n_rows}** pasien, hanya **{n_stroke}** yang mengalami stroke "
    f"(≈ {prevalence:.2f}%). Ini menunjukkan **kelas yang sangat tidak seimbang** "
    "sehingga metode penanganan imbalanced data (misalnya SMOTE) menjadi penting di tahap pre-processing."
)

# --------------------------------------------------
# 4) Deskripsi variabel
# --------------------------------------------------
st.markdown("### 📌 Deskripsi Variabel")

with st.expander("Lihat deskripsi lengkap variabel"):
    var_desc = [
        ["id", "Numeric (ID)", "Identitas unik tiap pasien"],
        ["gender", "Kategori", "Jenis kelamin pasien (Male/Female/Other)"],
        ["age", "Numeric", "Usia pasien dalam tahun"],
        ["hypertension", "Biner", "1 jika pasien memiliki riwayat hipertensi, 0 jika tidak"],
        ["heart_disease", "Biner", "1 jika pasien memiliki penyakit jantung, 0 jika tidak"],
        ["ever_married", "Kategori", "Status pernah menikah (Yes/No)"],
        ["work_type", "Kategori", "Jenis pekerjaan (Private, Self-employed, Govt_job, dll.)"],
        ["Residence_type", "Kategori", "Tipe tempat tinggal (Urban/Rural)"],
        ["avg_glucose_level", "Numeric", "Rata-rata kadar glukosa dalam darah"],
        ["bmi", "Numeric", "Body Mass Index (kg/m²), terdapat missing values"],
        ["smoking_status", "Kategori", "Status merokok (formerly smoked, never smoked, smokes, unknown)"],
        ["stroke", "Biner (Target)", "1 jika pasien pernah mengalami stroke, 0 jika tidak"]
    ]
    var_df = pd.DataFrame(var_desc, columns=["Variabel", "Tipe", "Keterangan"])
    st.dataframe(var_df, use_container_width=True)

st.markdown(
    """
Homepage ini memberikan konteks umum sebelum pengguna berpindah ke:
- **Data Problem & Analysis** → membahas imbalance, missing, dan outlier secara detail  
- **Pre-Processing Data** → menampilkan langkah transformasi data  
- **Feature Selection** → melihat fitur mana yang paling berpengaruh  
- **Modelling & Validation** → membandingkan performa model dan memilih model terbaik  
"""
)
