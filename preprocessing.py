import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
import os

nltk.data.path.append('/tmp')
nltk.download('stopwords', download_dir='/tmp')

# ======================
# load kamus normalisasi
# ====================== 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
normalize_df = pd.read_csv(
    os.path.join(BASE_DIR, 'data/singkatan-lib.csv'),
    header=None,
    names=["slang", "standard"]
)

normalization_dict = dict(
    zip(normalize_df["slang"].str.lower(), normalize_df["standard"].str.lower())
)

# ======================
# stopwords
# ======================
stop_words = set(stopwords.words('indonesian'))

# kata negasi yang harus dipertahankan
negation_word = {'tidak', 'bukan', 'tanpa', 'jangan', 'belum'}
# hapus negasi dari stopwords
stop_words = stop_words - negation_word

# custom stopwords 
custom_stopwords = {
    'hahahaha', 'haha', 'hehe', 'wkwk','bg', 'bang', 'sih', 'yah',
    'aja', 'nih', 'deh', 'loh', 'ku', 'ya', 'bro', 'bos', 'guys'
}

stop_words = stop_words.union(custom_stopwords)

# ======================
# stemming
# ======================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# ======================
# FUNCTION PIPELINE
# ======================

def cleaning(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # hapus URL
    text = re.sub(r"@\w+", "", text)                     # hapus mention
    text = re.sub(r"[^a-zA-Z\s]", " ", text)             # hapus angka & simbol
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)           # huruf berulang
    text = re.sub(r"\s+", " ", text).strip()             # spasi berlebih
    return text

def case_folding(text):
    return text.lower()

def tokenizing(text):
    return text.split()

def remove_duplicate_word(tokens):
    if not tokens:
        return tokens
    result = [tokens[0]]
    for word in tokens[1:]:
        if word != result[-1]:
            result.append(word)
    return result

def normalisasi(tokens):
    return [normalization_dict.get(word, word) for word in tokens]

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

def stemming(tokens):
    return [stemmer.stem(t) for t in tokens]

def preprocessing(text):
    text = cleaning(text)
    text = case_folding(text)
    tokens = tokenizing(text)
    tokens = normalisasi(tokens)
    tokens = remove_stopwords(tokens)
    tokens = stemming(tokens)
    return " ".join(tokens)