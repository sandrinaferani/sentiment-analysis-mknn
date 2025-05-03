import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# 1. Cleaning Function
def remove(text):
    text = ''.join([char if not char.isdigit() else ' ' for char in text])
    text = ''.join([char if char.isalnum() or char.isspace() else ' ' for char in text])
    text = ''.join([char if ord(char) < 128 else '' for char in text])
    return ' '.join(text.split())

# 2. Case Folding
def case_folding(text):
    return ''.join([chr(ord(char) + 32) if 'A' <= char <= 'Z' else char for char in text])

# 3. Tokenization
def tokenize(sentence):
    words, word = [], ''
    for char in sentence:
        if char != ' ':
            word += char
        else:
            if word:
                words.append(word)
                word = ''
    if word:
        words.append(word)
    return words

# 4. Normalization
def load_normalization_dict(path):
    df = pd.read_excel(path)
    return dict(zip(df['before'], df['after']))

def normalisasi(tokens, normalization_dict): 
    return [normalization_dict.get(word, word) for word in tokens]

# 5. Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(tokens):
    changes, stemmed_tokens = [], []
    for word in tokens:
        if not isinstance(word, str):
            word = str(word)
        stemmed_word = stemmer.stem(word)
        stemmed_tokens.append(stemmed_word)
        if word != stemmed_word:
            changes.append((word, stemmed_word))
    return stemmed_tokens

# 6. Stopwords Removal
def load_stopwords(path):
    df = pd.read_excel(path)
    return set(df['stopwords'].dropna())

def remove_stopwords(tokens, stopwords_set):
    return [token for token in tokens if token not in stopwords_set]

# Load resources
normalization_dict = load_normalization_dict("normalisasi.xlsx")
stopwords_set = load_stopwords("stopwords.xlsx")

# # Load and prepare data
# data = pd.read_excel("coba.xlsx")
# data['reviews'] = data['reviews'].fillna('').astype(str)

# # Apply preprocessing step by step
# data['cleaning'] = data['reviews'].apply(remove)
# data['case_folding'] = data['cleaning'].apply(case_folding)
# data['tokenization'] = data['case_folding'].apply(tokenize)
# data['normalization'] = data['tokenization'].apply(lambda x: normalisasi(x, normalization_dict))
# data['stemming'] = data['normalization'].apply(stemming)
# data['final'] = data['stemming'].apply(lambda x: remove_stopwords(x, stopwords_set))

# # Tampilkan hasil akhir
# print(data[['reviews', 'final']])
