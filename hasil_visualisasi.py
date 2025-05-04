import streamlit as st
import pandas as pd
from PIL import Image
import os

def tampilkan_hasil_statis():
    st.title("📊 Analisis Sentimen - April 2025")
    st.markdown("Perbandingan hasil sentimen untuk tiga aplikasi marketplace: **Lazada**, **Shopee**, dan **Tokopedia**.")

    # Pilihan minggu
    minggu_ke = st.selectbox("Pilih Minggu:", options=[1, 2, 3, 4], format_func=lambda x: f"Minggu ke-{x}")

    kolom = st.columns(3)
    apps = ["Lazada", "Shopee", "Tokopedia"]
    urutan_file = ["lazada", "shopee", "tokopedia"]

    for i, col in enumerate(kolom):
        app = apps[i]
        filekey = urutan_file[i]

        with col:
            st.subheader(f"{i+1}. {app}")

            # Pie Chart Persentase
            chart_path = f"user/persentase-2025-april-{minggu_ke}-{filekey}.png"
            if os.path.exists(chart_path):
                st.image(chart_path, caption="Distribusi Sentimen", use_column_width=True)
            else:
                st.warning("Grafik persentase tidak ditemukan.")

            # Expander: Tabel dan WordCloud
            with st.expander("Lihat Detail"):
                # Tabel hasil bisa dari .xlsx atau .csv
                tabel_excel = f"user/tabel-2025-april-{minggu_ke}-{filekey}.xlsx"
                tabel_csv = f"user/tabel-2025-april-{minggu_ke}-{filekey}.csv"
                df = None

                if os.path.exists(tabel_excel):
                    try:
                        df = pd.read_excel(tabel_excel)
                    except Exception as e:
                        st.error(f"Gagal membaca file Excel: {e}")
                elif os.path.exists(tabel_csv):
                    try:
                        df = pd.read_csv(tabel_csv)
                    except Exception as e:
                        st.error(f"Gagal membaca file CSV: {e}")
                else:
                    st.warning("Tabel hasil tidak ditemukan (xlsx/csv).")

                if df is not None:
                    df.columns = df.columns.str.strip()
                    if {'reviews', 'Prediksi'}.issubset(df.columns):
                        st.markdown("#### Hasil Klasifikasi")
                        st.dataframe(df[['reviews', 'Prediksi']], use_container_width=True)
                    else:
                        st.error("Kolom 'reviews' dan/atau 'Prediksi' tidak ditemukan.")
                        st.write("Kolom yang tersedia:", df.columns.tolist())

                # Wordcloud Sentimen
                st.markdown("#### Word Cloud")
                for sentimen in ["positif", "netral", "negatif"]:
                    wc_path = f"user/{sentimen}-2025-april-{minggu_ke}-{filekey}.png"
                    if os.path.exists(wc_path):
                        st.markdown(f"**{sentimen.capitalize()}**")
                        st.image(Image.open(wc_path), use_column_width=True)
                    else:
                        st.info(f"Tidak ada wordcloud untuk {sentimen}.")
