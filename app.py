from lib import *
from model import *
from preprocessing import *
from hasil_visualisasi import *

st.set_page_config(page_title="MKNN Sentiment Analysis", layout="wide")

# Navigasi Sidebar
st.sidebar.markdown("Menu")
mode = st.sidebar.radio("Pilih Menu", ["🔍 Prediksi Baru", "📊 Analisis Mingguan"])

# Tampilkan judul hanya jika bukan mode statis
if mode != "📊 Analisis Mingguan":
    st.title("Analisis Sentimen Aplikasi Marketplace🛍️")
    st.markdown("Selamat datang di aplikasi analisis sentimen untuk marketplace! \nUnggah satu atau lebih file ulasan (dalam format **CSV/XLSX**) dengan kolom `reviews`, lalu lihat perbandingan sentimen antar platform secara visual.")

# Mode Statis
if mode == "📊 Analisis Mingguan":
    tampilkan_hasil_statis()
    st.stop()


# Mode Prediksi Baru
@st.cache_data
def load_train_data():
    df_train = pd.read_excel("train_model_k1530%.xlsx")
    labels = df_train['label'].values
    selected_features = list(df_train.columns[1:-1])
    X_train = df_train[selected_features].values
    return X_train, labels, selected_features

X_train, y_train, selected_features = load_train_data()

uploaded_files = st.file_uploader(
    "Upload satu atau lebih file (CSV/XLSX dengan kolom 'reviews')",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    is_single_file = len(uploaded_files) == 1
    sort_by_sentimen = False
    if not is_single_file:
        sort_by_sentimen = st.checkbox("🔼 Peringkatkan")

    file_results = []

    for uploaded_file in uploaded_files:
        df_test = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        if 'reviews' not in df_test.columns:
            st.error(f"❌ Kolom 'reviews' tidak ditemukan di file {uploaded_file.name}")
            continue

        start_time = time.time()

        # Preprocessing
        with st.spinner('⚙️Tahap preproses data (1/3)...'):
            df_test['clean'] = df_test['reviews'].fillna('').astype(str).apply(remove).apply(case_folding)
            df_test['tokens'] = df_test['clean'].apply(tokenize)
            df_test['normalized'] = df_test['tokens'].apply(lambda x: normalisasi(x, normalization_dict))
            df_test['stemmed'] = df_test['normalized'].apply(stemming)
            df_test['final_tokens'] = df_test['stemmed'].apply(lambda x: remove_stopwords(x, stopwords_set))

        # TF-IDF
        with st.spinner('⚙️Tahap ekstraksi fitur (2/3)...'):
            vocab = selected_features
            token_lists = df_test['final_tokens'].tolist()
            tf_list = [compute_tf(tokens, vocab) for tokens in token_lists]
            idf = compute_idf(vocab, token_lists)
            tfidf_vectors = [compute_tfidf(tf, idf, vocab) for tf in tf_list]
            X_test = np.array([[vec[term] for term in vocab] for vec in tfidf_vectors])

        # MKNN
        with st.spinner('⚙️Tahap klasifikasi data (3/3)...'):
            k = 5
            alpha = 0.5
            validities = get_or_load_validities(X_train, y_train, k)
            y_pred = mknn(X_train, y_train, X_test, k=k, alpha=alpha, validities=validities)
            df_test['Prediksi'] = y_pred

        total_time = time.time() - start_time

        pred_counts = df_test['Prediksi'].value_counts()
        total = pred_counts.sum()
        persentase_positif = (pred_counts.get('positif', 0) / total) * 100 if total > 0 else 0
        persentase_netral = (pred_counts.get('netral', 0) / total) * 100 if total > 0 else 0
        persentase_negatif = (pred_counts.get('negatif', 0) / total) * 100 if total > 0 else 0

        file_results.append((uploaded_file.name, df_test, pred_counts,
                             persentase_positif, persentase_netral, persentase_negatif, total_time))

    # Urutkan
    if sort_by_sentimen:
        with st.spinner("🔄 Memperingkatkan Aplikasi"):
            file_results = sorted(
                file_results,
                key=lambda x: (
                    round(x[3], 4),
                    round(x[4], 4),
                    -round(x[5], 4)
                ),
                reverse=True
            )

    layout = st.columns(2) if not is_single_file else st.columns([1.5, 4.5, 1.5])

    for idx, (filename, df_test, pred_counts,
              persentase_positif, persentase_netral, persentase_negatif, total_time) in enumerate(file_results):

        display_area = layout[1] if is_single_file else (layout[0] if idx % 2 == 0 else layout[1])
        with display_area:
            base_filename, _ = os.path.splitext(filename)
            display_name = f"{idx+1}. {base_filename}" if sort_by_sentimen else base_filename
            if sort_by_sentimen:
                st.subheader(f"{display_name}")
            else:
                st.subheader(f"📄 File: {base_filename}")

            st.markdown(f"🕒 Waktu Proses: {total_time:.2f} detik")
            st.success("✅ Prediksi selesai!")

            # Pie Chart
            st.markdown("#### 📊 Statistik Prediksi")
            labels = pred_counts.index.tolist()
            sizes = pred_counts.values.tolist()
            explode = [0.05] * len(sizes)
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.pie(
                sizes,
                labels=[f"{label} ({count})" for label, count in zip(labels, sizes)],
                autopct='%1.1f%%',
                startangle=20,
                explode=explode,
                colors=['#ff9999', '#66b3ff', '#99ff99']
            )
            ax.axis('equal')
            st.pyplot(fig, use_container_width=True)


            with st.expander("📋Detail Hasil Klasifikasi"):
                st.markdown("#### Hasil Klasifikasi")
                st.dataframe(df_test[['reviews', 'Prediksi']], use_container_width=True)

                st.markdown("#### Word Cloud Berdasarkan Sentimen")

                def create_wordcloud(text, title):
                    if text.strip():
                        wordcloud = WordCloud(width=500, height=300, background_color='white', max_words=50).generate(text)
                        fig, ax = plt.subplots(figsize=(4, 2))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.subheader(f"{title}")
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info(f"Tidak ada data untuk {title}.")

                negative_reviews = ' '.join(df_test[df_test['Prediksi'].str.lower() == 'negatif']['final_tokens'].apply(lambda x: ' '.join(x)))
                positive_reviews = ' '.join(df_test[df_test['Prediksi'].str.lower() == 'positif']['final_tokens'].apply(lambda x: ' '.join(x)))
                neutral_reviews = ' '.join(df_test[df_test['Prediksi'].str.lower() == 'netral']['final_tokens'].apply(lambda x: ' '.join(x)))

                create_wordcloud(positive_reviews, "Positif")
                create_wordcloud(neutral_reviews, "Netral")
                create_wordcloud(negative_reviews, "Negatif")
