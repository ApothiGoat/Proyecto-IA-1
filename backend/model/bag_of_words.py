def build_vocabulary(tokenized_texts):
    """
    Construye el vocabulario (todas las palabras únicas).
    
    Entrada:
    - tokenized_texts: lista de listas de tokens
    
    Salida:
    - vocab: diccionario {palabra: índice}
    """
    
    vocab = {}
    
    for tokens in tokenized_texts:
        for word in tokens:
            if word not in vocab:
                vocab[word] = len(vocab)
    
    return vocab

def vectorize(tokens, vocab):
    """
    Convierte una lista de tokens en un vector de frecuencias.
    
    Entrada:
    - tokens: lista de palabras
    - vocab: diccionario de palabras
    
    Salida:
    - vector: lista de números
    """
    
    vector = [0] * len(vocab)
    
    for word in tokens:
        if word in vocab:
            vector[vocab[word]] += 1
    
    return vector

def vectorize_dataset(tokenized_texts, vocab):
    """
    Convierte todos los textos en vectores.
    """
    
    return [vectorize(tokens, vocab) for tokens in tokenized_texts]