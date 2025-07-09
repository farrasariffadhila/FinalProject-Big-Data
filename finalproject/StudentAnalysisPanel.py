import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Student Analysis Panel", layout="wide")
st.title("📊 Student Analysis Panel")

# Pilih file data utama
available_files = [f for f in ["clustered_students.csv", "clustered_students_scaled.csv"] if os.path.exists(f)]
if not available_files:
    st.error("Tidak ada file data yang ditemukan. Pastikan file CSV sudah ada di folder ini.")
    st.stop()

data_file = st.sidebar.selectbox("Pilih file data utama (untuk visualisasi utama):", available_files)
st.markdown(f"**Data utama yang digunakan:** `{data_file}`")

# Load data utama
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

df = load_data(data_file)

# Otomatis mapping cluster ke label jika hanya ada kolom 'cluster'
if 'cluster_label' not in df.columns and 'cluster' in df.columns:
    label_map = {0: "Diligent", 1: "Lazy"}
    df['cluster_label'] = df['cluster'].map(label_map).fillna(df['cluster'].astype(str))

# Deteksi kolom cluster label
label_col = None
for col in ['cluster_label', 'label', 'Cluster', 'cluster']:
    if col in df.columns:
        label_col = col
        break
if not label_col:
    st.error("Kolom label cluster tidak ditemukan di data utama.")
    st.stop()

# Sidebar filter cluster
cluster_options = df['cluster_label'].unique().tolist()
selected_cluster = st.sidebar.multiselect("Cluster", cluster_options, default=cluster_options)
filtered_df = df[df['cluster_label'].isin(selected_cluster)]

# Info data
st.info(f"Jumlah data: {len(filtered_df)} | Jumlah fitur: {len(df.columns)} | Cluster: {cluster_options}")

# Download button
st.download_button(
    label="Download data utama yang difilter",
    data=filtered_df.to_csv(index=False),
    file_name=f"filtered_{data_file}",
    mime="text/csv"
)

# Deteksi fitur numerik
exclude_cols = [label_col, 'PCA1', 'PCA2']
numeric_features = [col for col in df.select_dtypes(include='number').columns if col not in exclude_cols]

# Pie chart
st.subheader("Proporsi Siswa per Cluster")
cluster_counts = filtered_df['cluster_label'].value_counts().reset_index()
cluster_counts.columns = ['cluster_label', 'count']
fig_pie = px.pie(cluster_counts, names='cluster_label', values='count', title='Proporsi Cluster', color_discrete_sequence=px.colors.qualitative.Set1)
st.plotly_chart(fig_pie, use_container_width=True)

# Radar chart
if len(numeric_features) > 1:
    st.subheader("Perbandingan Rata-rata Fitur Numerik per Cluster (Radar Chart)")
    radar_data = filtered_df.groupby('cluster_label')[numeric_features].mean().reset_index()
    fig_radar = go.Figure()
    for i, row in radar_data.iterrows():
        fig_radar.add_trace(go.Scatterpolar(r=row[numeric_features], theta=numeric_features, fill='toself', name=row['cluster_label']))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Fitur numerik untuk radar chart tidak ditemukan atau hanya satu fitur.")

# =======================
# Bagian dari data asli (unscaled)
# =======================
if "clustered_students.csv" in available_files:
    st.markdown("---")
    st.header("📋 Statistik & Distribusi dari Data Asli (Unscaled)")

    df_raw = load_data("clustered_students.csv")
    # Mapping label jika perlu
    if 'cluster_label' not in df_raw.columns and 'cluster' in df_raw.columns:
        label_map = {0: "Diligent", 1: "Lazy"}
        df_raw['cluster_label'] = df_raw['cluster'].map(label_map).fillna(df_raw['cluster'].astype(str))

    # Filter cluster sesuai pilihan di sidebar
    filtered_df_raw = df_raw[df_raw['cluster_label'].isin(selected_cluster)]

    # Deteksi fitur numerik data asli
    label_col_raw = 'cluster_label'
    exclude_cols_raw = [label_col_raw, 'PCA1', 'PCA2']
    numeric_features_raw = [col for col in df_raw.select_dtypes(include='number').columns if col not in exclude_cols_raw]

    # Ringkasan Statistik per Cluster (data asli)
    st.subheader("Ringkasan Statistik per Cluster (Data Asli)")
    st.dataframe(filtered_df_raw.groupby('cluster_label')[numeric_features_raw].mean().T.style.format("{:.2f}"))

    # Boxplot (data asli)
    if len(numeric_features_raw) > 0:
        st.subheader("Distribusi Fitur Numerik per Cluster (Boxplot)")
        selected_feature_raw = st.selectbox("Pilih fitur untuk boxplot:", numeric_features_raw)
        fig_box_raw = px.box(filtered_df_raw, x='cluster_label', y=selected_feature_raw, color='cluster_label', title=f'Distribusi {selected_feature_raw} per Cluster (Data Asli)', color_discrete_sequence=px.colors.qualitative.Set1)
        st.plotly_chart(fig_box_raw, use_container_width=True)
else:
    st.warning("File clustered_students.csv (data asli) tidak ditemukan. Statistik dan boxplot data asli tidak bisa ditampilkan.")
