import re  # Librería para expresiones regulares
import nltk  # Librería de NLP
from nltk.corpus import stopwords  # Lista de palabras comunes
from nltk.stem import PorterStemmer  # Algoritmo para reducir palabras a su raíz

# Descargar recursos necesarios de NLTK (solo la primera vez)
nltk.download('stopwords')
# Crear conjunto de stopwords en ingles
stop_words = set(stopwords.words('english'))

# Inicializar el stemmer (para reducir palabras a su raíz)
stemmer = PorterStemmer()


def clean_text(text):
    """
    Esta función recibe un texto y devuelve una lista de palabras limpias (tokens)
    """

    # 1. Convertir todo a minúsculas
    text = text.lower()
    
    # 2. Eliminar todo lo que no sea letras o espacios
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenización (separar en palabras)
    tokens = text.split()
    
    # 4. Eliminar stopwords (palabras sin valor semántico)
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Stemming (reducir palabras a su raíz)
    # Ej: "running", "runs", "ran" → "run"
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Retorna lista de palabras procesadas
    return tokens