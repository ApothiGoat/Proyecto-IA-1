import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descarga las stopwords si no existen
nltk.download("stopwords")

# Stopwords en inglés
stop_words = set(stopwords.words("english"))

# Stemmer para reducir palabras a su raíz
stemmer = PorterStemmer()


def clean_text(text):
    """
    Limpia una solicitud de cliente y devuelve una lista de tokens procesados.
    """

    # Convertir a string por seguridad
    text = str(text)

    # Eliminar placeholders tipo {{Order Number}}, {{Website URL}}, etc.
    text = re.sub(r"\{\{.*?\}\}", " ", text)

    # Convertir a minúsculas
    text = text.lower()

    # Eliminar caracteres que no sean letras o espacios
    text = re.sub(r"[^a-z\s]", " ", text)

    # Eliminar espacios repetidos
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenizar separando por espacios
    tokens = text.split()

    # Eliminar stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Aplicar stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens