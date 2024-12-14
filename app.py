import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

# Pengaturan halaman (harus di bagian atas)
st.set_page_config(page_title="Aplikasi Klasifikasi Sampah", page_icon="🗑️", layout="centered", initial_sidebar_state="expanded")

# Fungsi untuk memuat model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('/content/drive/MyDrive/dataset3/.mdl_wt.keras')
    return model

model = load_model()

# Fungsi untuk prediksi gambar
def import_and_predict(image_data, model):
    size = (224, 224)  # Ubah ukuran menjadi (224, 224)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

st.markdown("""
    <style>
        /* Desain kartu tim */
        .team-card {
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s ease-in-out;
            text-align: center;
            width: 200px;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .team-card:hover {
            transform: scale(1.05);
            box-shadow: 0px 6px 14px rgba(0, 0, 0, 0.15);
        }
        .team-card h1 {
            font-size: 18px;
            color: #1a2634; /* Warna biru navy tua */
        }
    </style>
""", unsafe_allow_html=True)

# Halaman utama
st.title("Aplikasi Klasifikasi Sampah")
st.sidebar.title("Navigasi")
option = st.sidebar.radio("Pilih Halaman", ("Tim", "Penjelasan", "Klasifikasi"))

# Halaman Tim
if option == "Tim":
    st.title("Our Team")
    st.write("Meet the brilliant minds behind this project!")
    col1, col2, col3 = st.columns(3)

    with col1:
        image_path = "/content/drive/MyDrive/dataset3/foto/euis.jpg"
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True, width=150)
        except FileNotFoundError:
            st.error("File gambar tidak ditemukan pada path yang ditentukan.")

        st.markdown("""
            <div class="team-card">
                <p><strong>Euis Nurhanifah</strong></p>
                <p>1217050046</p>
                <p>Laporan</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        image_path = "/content/drive/MyDrive/dataset3/foto/kensa.jpg"
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True, width=150)
        except FileNotFoundError:
            st.error("File gambar tidak ditemukan pada path yang ditentukan.")

        st.markdown("""
            <div class="team-card">
                <p><strong>Kensa Baeren Deftnamor</strong></p>
                <p>1217050071</p>
                <p>Pemodelan</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        image_path = "/content/drive/MyDrive/dataset3/foto/kirei.jpg"
        try:
            image = Image.open(image_path)
            st.image(image, use_container_width=True, width=150)
        except FileNotFoundError:
            st.error("File gambar tidak ditemukan pada path yang ditentukan.")

        st.markdown("""
            <div class="team-card">
                <p><strong>Kireina Amani Ridiesto</strong></p>
                <p>1217050074</p>
                <p>Interface</p>
            </div>
        """, unsafe_allow_html=True)

# Halaman Penjelasan
elif option == "Penjelasan":
    st.header("Tentang Aplikasi")
    image_path = "/content/drive/MyDrive/dataset3/foto/penjelasan.jpg"
    try:
        image = Image.open(image_path)
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.error("File gambar tidak ditemukan pada path yang ditentukan.")

    # Tambahkan teks deskripsi di bawah gambar
    st.write("""
    Sampah merupakan salah satu masalah global yang berdampak pada lingkungan dan kesehatan masyarakat. Pengelolaan sampah yang tidak efektif, seperti sulitnya memisahkan jenis sampah, menyebabkan masalah lebih lanjut, seperti pencemaran lingkungan.

    Aplikasi ini bertujuan membantu masyarakat, lembaga daur ulang, dan instansi terkait untuk mengidentifikasi jenis sampah secara cepat dan akurat. Dengan klasifikasi otomatis, pengelompokan sampah menjadi lebih mudah, mendukung proses daur ulang.

    **Kategori Sampah yang Didukung:**
    - Kardus (Cardboard)
    - Logam (Metal)
    - Kertas (Paper)
    - Plastik (Plastic)
    - Sampah lainnya (Trash)
    """)

# Halaman Klasifikasi
elif option == "Klasifikasi":
    st.header("Klasifikasi Sampah")
    image_path = "/content/drive/MyDrive/dataset3/foto/klasifikasi.jpg"
    try:
        image = Image.open(image_path)
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.error("File gambar tidak ditemukan pada path yang ditentukan.")
    # Tambahkan teks deskripsi di bawah gambar

    file = st.file_uploader("Silakan unggah gambar sampah", type=["jpg", "png"])
    if file is not None:
        image = Image.open(file)
        st.image(image, use_container_width=True)

        # Prediksi
        predictions = import_and_predict(image, model)
        class_names = ['kardus', 'logam', 'kertas', 'plastik', 'sampah']
        predicted_class = class_names[np.argmax(predictions)]
        st.success(f"Gambar ini kemungkinan besar adalah: {predicted_class}")
