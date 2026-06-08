# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go  # Pastikan library plotly sudah di-install

# ==========================================
# KONFIGURASI HALAMAN UTAMA
# ==========================================
st.set_page_config(page_title="Airline Satisfaction App", layout="wide", page_icon="✈️")

# ==========================================
# CUSTOM CSS: Background Cream & Perbaikan Input
# ==========================================
st.markdown(
    """
    <style>
    /* 1. Mengubah background utama aplikasi menjadi cream */
    .stApp {
        background-color: #FAF3E0 !important; 
    }
    
    /* 2. Memaksa teks judul dan label menjadi warna hitam */
    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #1E1E1E !important;
    }
    
    /* 3. PERBAIKAN INPUT: Background kotak input & Dropdown jadi putih, teks hitam */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"],
    input[type="number"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 5px !important;
    }

    /* Memaksa teks ketikan di dalam input menjadi hitam */
    input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Memaksa isi dan panah dropdown berwarna hitam */
    div[data-baseweb="select"] * {
        color: #000000 !important;
    }

    /* 4. PERBAIKAN POPOVER (Daftar list saat dropdown diklik) */
    div[data-baseweb="popover"] div {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    
    /* 5. Mempercantik tombol prediksi (Warna Hijau) */
    button[kind="primary"] {
        background-color: #8B6F47 !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# KONFIGURASI GRAFIK AGAR TIDAK HITAM
# ==========================================
plt.rcParams['figure.facecolor'] = '#FAF3E0'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['text.color'] = '#1E1E1E'
plt.rcParams['axes.labelcolor'] = '#1E1E1E'
plt.rcParams['xtick.color'] = '#1E1E1E'
plt.rcParams['ytick.color'] = '#1E1E1E'
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#FFFFFF", "figure.facecolor":"#FAF3E0"})

# ==========================================
# MEMUAT MODEL DAN DATA
# ==========================================
@st.cache_resource
def load_model_artifacts():
    return joblib.load('model_artifacts.pkl')

try:
    artifacts = load_model_artifacts()
    model = artifacts['model']
    encoders = artifacts['encoders']
    imputer = artifacts['imputer']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
except FileNotFoundError:
    st.error("File 'model_artifacts.pkl' tidak ditemukan. Sila jalankan 'python train.py' terlebih dahulu di terminal Anda.")
    st.stop()

@st.cache_data
def load_raw_data():
    data = pd.read_csv('test.csv', sep=';')
    if 'Column1' in data.columns: data.drop(['Column1'], axis=1, inplace=True)
    if 'id' in data.columns: data.drop(['id'], axis=1, inplace=True)
    return data

df_raw = load_raw_data()

# ==========================================
# JUDUL APLIKASI
# ==========================================
st.title("✈️ Klasifikasi Kepuasan Penumpang Maskapai")
st.markdown("Sistem prediktif berbasis *Machine Learning* untuk menganalisis dan mengklasifikasikan kepuasan layanan penerbangan secara cepat dan akurat.")
st.markdown("---")

# ==========================================
# TAB NAVIGASI
# ==========================================
tab_prediksi, tab_eda = st.tabs(["📋 Formulir Prediksi (Klasifikasi)", "📊 Dashboard Visualisasi (EDA)"])

# ==========================================
# TAB 1: FORM INPUT
# ==========================================
with tab_prediksi:
    st.subheader("Silakan lengkapi parameter penilaian layanan di bawah ini:")
    
    with st.form("form_klasifikasi"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Profil Demografi & Penerbangan**")
            gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
            customer_type = st.selectbox("Tipe Pelanggan", ["Loyal Customer", "disloyal Customer"])
            age = st.slider("Umur Penumpang", 7, 85, 35)
            type_of_travel = st.selectbox("Tujuan Perjalanan", ["Business travel", "Personal Travel"])
            flight_class = st.selectbox("Kelas Penerbangan", ["Business", "Eco", "Eco Plus"])
            flight_distance = st.number_input("Jarak Penerbangan (mil)", min_value=10, max_value=5000, value=1000)

        with col2:
            st.markdown("**🛫 Kualitas Layanan Pesawat (Rating 0-5)**")
            wifi = st.selectbox("Koneksi Wifi Pesawat", [0,1,2,3,4,5], index=3)
            seat_comfort = st.selectbox("Kenyamanan Kursi", [0,1,2,3,4,5], index=3)
            food_drink = st.selectbox("Kualitas Makanan & Minuman", [0,1,2,3,4,5], index=3)
            entertainment = st.selectbox("Hiburan di Dalam Pesawat", [0,1,2,3,4,5], index=3)
            on_board_service = st.selectbox("Pelayanan Kru Kabin", [0,1,2,3,4,5], index=3)
            leg_room = st.selectbox("Kelegaan Ruang Kaki", [0,1,2,3,4,5], index=3)
            cleanliness = st.selectbox("Kebersihan Pesawat", [0,1,2,3,4,5], index=3)

        with col3:
            st.markdown("**🛬 Layanan Bandara & Keterlambatan**")
            online_booking = st.selectbox("Kemudahan Pemesanan Online", [0,1,2,3,4,5], index=3)
            time_convenient = st.selectbox("Kemudahan Jam Penerbangan", [0,1,2,3,4,5], index=3)
            online_boarding = st.selectbox("Proses Online Boarding", [0,1,2,3,4,5], index=3)
            checkin = st.selectbox("Pelayanan Check-in", [0,1,2,3,4,5], index=3)
            baggage = st.selectbox("Penanganan Bagasi", [1,2,3,4,5], index=3)
            gate_location = st.selectbox("Lokasi Gerbang (Gate)", [0,1,2,3,4,5], index=3)
            inflight_service = st.selectbox("Pelayanan Ground/Inflight Umum", [0,1,2,3,4,5], index=3)
            
            st.markdown("**(Keterlambatan dalam Menit)**")
            col3_1, col3_2 = st.columns(2)
            with col3_1:
                dep_delay = st.number_input("Delay Berangkat", min_value=0, max_value=1500, value=0)
            with col3_2:
                arr_delay = st.number_input("Delay Tiba", min_value=0, max_value=1500, value=0)

        # Tombol Submit Form
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Proses Klasifikasi Kepuasan", type="primary", use_container_width=True)

    # =========================================================
    # LOGIKA PREDIKSI (Berjalan HANYA jika tombol ditekan)
    # =========================================================
    if submitted:
        # 1. Prapemrosesan Data
        input_data = pd.DataFrame([{
            'Gender': gender, 'Customer Type': customer_type, 'Age': age,
            'Type of Travel': type_of_travel, 'Class': flight_class, 'Flight Distance': flight_distance,
            'Inflight wifi service': wifi, 'Departure/Arrival time convenient': time_convenient,
            'Ease of Online booking': online_booking, 'Gate location': gate_location,
            'Food and drink': food_drink, 'Online boarding': online_boarding, 'Seat comfort': seat_comfort,
            'Inflight entertainment': entertainment, 'On-board service': on_board_service,
            'Leg room service': leg_room, 'Baggage handling': baggage, 'Checkin service': checkin,
            'Inflight service': inflight_service, 'Cleanliness': cleanliness,
            'Departure Delay in Minutes': dep_delay, 'Arrival Delay in Minutes': arr_delay
        }])
        
        for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class']:
            input_data[col] = encoders[col].transform(input_data[col])
            
        input_data = input_data[feature_names]
        input_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_imputed)
        
        # 2. Prediksi Model
        prediction_code = model.predict(input_scaled)[0]
        prediction_label = encoders['satisfaction'].inverse_transform([prediction_code])[0]
        max_prob = max(model.predict_proba(input_scaled)[0]) * 100
        
        # 3. Kalkulasi Historis
        total_semua_data = len(df_raw)
        jumlah_target_sama = len(df_raw[df_raw['satisfaction'] == prediction_label])
        persentase_target = (jumlah_target_sama / total_semua_data) * 100
        
        # 4. RENDER HASIL KE DALAM KOTAK PUTIH
        st.markdown("---")
        st.subheader("📋 Hasil Klasifikasi & Analisis Data")

        res_col1, res_col2 = st.columns([1.5, 1])
        
        # Bagian Kiri: Kotak Putih berisi Teks Hasil & Historis
        with res_col1:
            if prediction_label == "satisfied":
                status_color = "#4CAF50"
                status_text = "PUAS (Satisfied)"
                status_desc = "Sistem cerdas kami memprediksi bahwa kombinasi layanan yang diterima oleh penumpang ini menghasilkan status <b>Sangat Puas</b>."
            else:
                status_color = "#E53935"
                status_text = "NETRAL / TIDAK PUAS"
                status_desc = "Sistem cerdas kami memprediksi bahwa kombinasi pelayanan maskapai yang diterima masuk dalam kategori <b>Netral atau Tidak Puas</b>."

            st.markdown(f"""
            <div style="background-color: #FFFFFF; padding: 25px; border-radius: 12px; border: 1px solid #E0E0E0; box-shadow: 2px 4px 10px rgba(0,0,0,0.05); height: 100%;">
                <p style="color: gray; font-size: 14px; margin-bottom: 5px; font-weight: bold;">HASIL PREDIKSI SAAT INI</p>
                <h1 style="color: {status_color}; margin: 0; font-size: 36px;">{status_text}</h1>
                <p style="font-size: 16px; margin-top: 10px; color: #1E1E1E;">{status_desc}</p>
                <hr style="border: 0.5px solid #EEEEEE; margin: 20px 0;">
                <p style="color: #1E1E1E; font-size: 16px; margin-bottom: 5px; font-weight: bold;">Data Historis Maskapai</p>
                <p style="font-size: 15px; color: #2B2B2B; line-height: 1.6; margin:0;">
                    Kategori profil ini mencakup <b>{persentase_target:.1f}%</b> dari populasi. 
                    Setara dengan <b>{jumlah_target_sama:,}</b> penumpang dari total keseluruhan {total_semua_data:,} data survei historis.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Bagian Kanan: Kotak Putih berisi Grafik Gauge
        with res_col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = max_prob,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Tingkat Keyakinan Model", 'font': {'size': 18, 'color': '#1E1E1E', 'weight': 'bold'}},
                number = {'suffix': "%", 'font': {'size': 35, 'color': '#1E1E1E'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#1E1E1E"},
                    'bar': {'color': "#1E1E1E", 'thickness': 0.25}, 
                    'bgcolor': "#F5F5F5",
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 50], 'color': '#FFCDD2'},    # Area Merah pudar
                        {'range': [50, 80], 'color': '#FFF9C4'},   # Area Kuning pudar
                        {'range': [80, 100], 'color': '#C8E6C9'}]  # Area Hijau pudar
                }
            ))
            
            # Memasukkan grafik ke dalam kotak putih yang serasi
            fig.update_layout(
                paper_bgcolor='#FFFFFF', 
                plot_bgcolor='#FFFFFF', 
                height=280, 
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.markdown("""
            <div style="background-color: #FFFFFF; padding: 10px; border-radius: 12px; border: 1px solid #E0E0E0; box-shadow: 2px 4px 10px rgba(0,0,0,0.05);">
            """, unsafe_allow_html=True)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: TOOL GRAFIK EDA
# ==========================================
with tab_eda:
    st.subheader("📊 Dashboard Analisis Grafik Interaktif")
    
    pilihan_grafik = st.selectbox("Pilih Sudut Pandang Data yang Ingin Divisualisasikan:", [
        "1. Distribusi Target Utama (Kepuasan Keseluruhan)",
        "2. Profil Demografi & Penerbangan (Kategorikal)",
        "3. Penilaian Kualitas Layanan Fasilitas (Rating 0-5)",
        "4. Distribusi Angka Keterlambatan & Jarak (Numerik)"
    ])
    
    st.markdown("---")
    
    cat_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    rating_features = ['Inflight wifi service', 'Departure/Arrival time convenient', 
                       'Ease of Online booking', 'Gate location', 'Food and drink', 
                       'Online boarding', 'Seat comfort', 'Inflight entertainment', 
                       'On-board service', 'Leg room service', 'Baggage handling', 
                       'Checkin service', 'Inflight service', 'Cleanliness']
    num_features = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']

    if pilihan_grafik.startswith("1"):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='satisfaction', data=df_raw, palette='Set2', ax=ax, edgecolor='black')
        plt.title("Perbandingan Jumlah Total Penumpang Puas vs Netral", fontweight='bold')
        st.pyplot(fig)
        
    elif pilihan_grafik.startswith("2"):
        fitur_pilihan = st.selectbox("Filter Berdasarkan Fitur:", cat_features)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(x=fitur_pilihan, hue='satisfaction', data=df_raw, palette='viridis', ax=ax, edgecolor='black')
        plt.title(f"Pengaruh {fitur_pilihan} Terhadap Kepuasan", fontweight='bold')
        st.pyplot(fig)
        
    elif pilihan_grafik.startswith("3"):
        fitur_pilihan_rating = st.selectbox("Pilih Fasilitas yang Ingin Dianalisis:", rating_features)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.countplot(x=fitur_pilihan_rating, hue='satisfaction', data=df_raw, palette='muted', ax=ax, edgecolor='black')
        plt.title(f"Analisis Kepuasan Berdasarkan Skor: {fitur_pilihan_rating}", fontweight='bold')
        st.pyplot(fig)
        
    elif pilihan_grafik.startswith("4"):
        fitur_pilihan_num = st.selectbox("Pilih Metrik Variabel Kontinu:", num_features)
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(x='satisfaction', y=fitur_pilihan_num, data=df_raw, palette='pastel', ax=ax)
        plt.title(f"Penyebaran Data {fitur_pilihan_num} terhadap Kepuasan", fontweight='bold')
        st.pyplot(fig)
