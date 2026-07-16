import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Klasifikasi Risiko Diabetes", layout="wide")

st.title("Klasifikasi Risiko Diabetes Pasien")
st.markdown("""
**Menggunakan Algoritma Decision Tree dan K-Nearest Neighbors (K-NN)**  
Kelompok 6 - Praktikum Data Mining | Universitas Pelita Bangsa
""")

with st.sidebar:
    st.header("Pengaturan")
    option = st.radio("Pilih sumber data:", ("Gunakan Dataset Default (Pima Indians)", "Upload Dataset sendiri"))
    uploaded_file = None
    if option == "Upload Dataset sendiri":
        uploaded_file = st.file_uploader("Upload file CSV", type="csv")

    st.subheader("Parameter Model")
    algo = st.selectbox("Pilih Algoritma", ["Decision Tree", "K-NN"])
    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100

    if algo == "Decision Tree":
        max_depth = st.slider("Max Depth", 1, 20, 5)
    else:
        cari_k = st.checkbox("Cari K Terbaik Otomatis")
        if cari_k:
            st.caption("K akan dicari dari 1–21 (ganjil)")
            k = 7
        else:
            k = st.slider("Nilai K", 1, 21, 7, 2)

    run = st.button("Jalankan Klasifikasi", type="primary")

@st.cache_data
def load_default_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
    df = pd.read_csv(url, names=cols)
    return df

def preprocess(df):
    df_clean = df.copy()
    cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in cols_with_zero:
        df_clean[col] = df_clean[col].replace(0, np.nan)
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    return df_clean

if run:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        expected_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                         "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            st.error(f"Dataset harus memiliki kolom: {', '.join(expected_cols)}. Kolom yang hilang: {', '.join(missing)}")
            st.stop()
        df = df[expected_cols]
    else:
        df = load_default_data()

    df_clean = preprocess(df)
    X = df_clean.drop("Outcome", axis=1)
    y = df_clean["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if algo == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        X_test_used = X_test
        feature_importances = model.feature_importances_
    else:
        if cari_k:
            k_range = range(1, 22, 2)
            k_scores = []
            for k_val in k_range:
                knn = KNeighborsClassifier(n_neighbors=k_val, metric="euclidean")
                knn.fit(X_train_scaled, y_train)
                k_scores.append(accuracy_score(y_test, knn.predict(X_test_scaled)))
            best_idx = np.argmax(k_scores)
            k = k_range[best_idx]
            st.success(f"K terbaik: **{k}** dengan akurasi **{k_scores[best_idx]:.2%}**")
            fig_k, ax_k = plt.subplots(figsize=(8, 4))
            ax_k.plot(list(k_range), k_scores, marker="o", linestyle="-", color="#2196F3")
            ax_k.axvline(x=k, color="red", linestyle="--", label=f"K terbaik = {k}")
            ax_k.set_xlabel("Nilai K")
            ax_k.set_ylabel("Akurasi")
            ax_k.set_title("Akurasi vs Nilai K")
            ax_k.set_xticks(list(k_range))
            ax_k.legend()
            st.pyplot(fig_k)
        model = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        X_test_used = X_test

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Tidak Diabetes", "Diabetes"], output_dict=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{acc:.2%}")
    col2.metric("Precision", f"{prec:.2%}")
    col3.metric("Recall", f"{rec:.2%}")
    col4.metric("F1-Score", f"{f1:.2%}")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Tidak Diabetes", "Diabetes"],
                    yticklabels=["Tidak Diabetes", "Diabetes"], ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

    with col_b:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.2f}", subset=report_df.select_dtypes(include=np.number).columns), use_container_width=True)

    if algo == "Decision Tree":
        st.subheader("Feature Importance")
        fi_df = pd.DataFrame({"Fitur": X.columns, "Importance": feature_importances}).sort_values("Importance", ascending=False)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))
        bars = ax2.barh(fi_df["Fitur"], fi_df["Importance"], color=colors)
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Fitur")
        ax2.invert_yaxis()
        for bar, val in zip(bars, fi_df["Importance"]):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.2%}", va="center", fontsize=9)
        st.pyplot(fig2)

    with st.expander("Lihat Data"):
        tab1, tab2, tab3 = st.tabs(["Data Awal", "Data Setelah Preprocessing", "Distribusi Kelas"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Total: {len(df)} baris, {len(df.columns)} kolom")
        with tab2:
            st.dataframe(df_clean.describe(), use_container_width=True)
        with tab3:
            class_dist = df_clean["Outcome"].value_counts().reset_index()
            class_dist.columns = ["Kelas", "Jumlah"]
            class_dist["Kelas"] = class_dist["Kelas"].map({0: "Tidak Diabetes", 1: "Diabetes"})
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            colors = ["#4CAF50", "#FF5722"]
            ax3.bar(class_dist["Kelas"], class_dist["Jumlah"], color=colors, width=0.5)
            ax3.set_ylabel("Jumlah")
            for i, v in enumerate(class_dist["Jumlah"]):
                ax3.text(i, v + 5, str(v), ha="center", fontweight="bold")
            st.pyplot(fig3)
else:
    st.info("Atur parameter di sidebar dan klik **Jalankan Klasifikasi** untuk memulai.")

st.markdown("---")
st.markdown("**Kelompok 6** | Gusti Ardhya Nanda Fahreza (312310624) · Dhiyaulhaq Al Maududi (312310508) · Robertus Amuala Alfonsius Daeli (312310634)")
