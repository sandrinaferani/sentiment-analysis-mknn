import numpy as np
import math
import os
# Tokenization function
def tokenize(doc):
    return [word.strip() for word in doc.strip("[]").replace("'", "").split(", ")]

# --- TF-IDF ---
def compute_tf(doc, vocabulary):
    n = len(doc)
    if n == 0:  # Jika panjang dokumen 0, hindari pembagian dengan nol
        return {term: 0.0 for term in vocabulary}  # Mengembalikan 0 untuk setiap term
    return {term: doc.count(term) / n for term in vocabulary}

def compute_idf(vocabulary, documents):
    N = len(documents)
    idf = {}
    for term in vocabulary:
        df = sum(1 for doc in documents if term in doc)
        idf[term] = math.log((N + 1) / (df + 1)) + 1
    return idf

def compute_tfidf(tf, idf, vocabulary):
    return {term: tf[term] * idf[term] for term in vocabulary}

# --- MKNN ---
def cosine_similarity_vector(v, w):
    return np.dot(v, w) / (np.linalg.norm(v) * np.linalg.norm(w) + 1e-8)

def calculate_validity(training_data, training_labels, k):
    n = len(training_data)
    validities = np.zeros(n)
    print(f"🔄 Menghitung validitas untuk {n} data latih...")
    for i in range(n):
        distances = [
            (j, cosine_similarity_vector(training_data[i], training_data[j]) if i != j else -1)
            for j in range(n)
        ]
        sorted_neighbors = sorted(distances, key=lambda x: x[1], reverse=True)[:k]
        neighbors_indices = [idx for idx, _ in sorted_neighbors]
        count_same_class = sum(1 for idx in neighbors_indices if training_labels[idx] == training_labels[i])
        validities[i] = count_same_class / k
        if i % 50 == 0 or i == n - 1:
            print(f"✅ Validity {i + 1}/{n} selesai.")
    return validities

def get_or_load_validities(X_train, y_train, k, filename="validities.npy"):
    if os.path.exists(filename):
        print("[CACHE] Memuat validitas dari cache...")
        return np.load(filename)
    else:
        validities = calculate_validity(X_train, y_train, k)
        np.save(filename, validities)
        print("[CACHE] Validitas disimpan ke cache.")
        return validities
    
# Di dalam model4.py
def mknn(training_data, training_labels, test_data, k, alpha, validities):
    validities = validities  # Menggunakan validitas yang sudah dihitung sebelumnya
    predictions = []

    for test_instance in test_data:
        distances = [(j, cosine_similarity_vector(test_instance, training_data[j])) for j in range(len(training_data))]
        sorted_neighbors = sorted(distances, key=lambda x: x[1], reverse=True)[:k]
        neighbors_indices = [idx for idx, _ in sorted_neighbors]

        weighted_votes = {}
        for idx in neighbors_indices:
            sim = dict(distances)[idx]
            weight = validities[idx] * (1 / (sim + alpha))
            label = training_labels[idx]
            weighted_votes[label] = weighted_votes.get(label, 0) + weight

        predicted_class = max(weighted_votes, key=weighted_votes.get)
        predictions.append(predicted_class)
    return predictions
