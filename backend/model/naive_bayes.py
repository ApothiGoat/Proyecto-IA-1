import math
from collections import defaultdict


class NaiveBayes:

    def __init__(self):
        self.class_probs = {}              # P(c)
        self.word_probs = {}               # P(w|c)
        self.class_word_counts = {}        # conteo palabras por clase
        self.class_totals = {}             # total palabras por clase
        self.vocab_size = 0
        self.classes = []

    def train(self, X, y):
        """
        Entrena el modelo.
        X: lista de vectores
        y: lista de etiquetas
        """

        self.classes = list(set(y))
        self.vocab_size = len(X[0])

        doc_count = len(y)

        # Inicializar estructuras
        self.class_word_counts = {c: [0]*self.vocab_size for c in self.classes}
        self.class_totals = {c: 0 for c in self.classes}
        class_doc_counts = defaultdict(int)

        # Contar
        for vector, label in zip(X, y):
            class_doc_counts[label] += 1

            for i, count in enumerate(vector):
                self.class_word_counts[label][i] += count
                self.class_totals[label] += count

        # Probabilidades a priori P(c)
        self.class_probs = {
            c: class_doc_counts[c] / doc_count
            for c in self.classes
        }

        # Probabilidades condicionales P(w|c) con Laplace
        self.word_probs = {}

        for c in self.classes:
            self.word_probs[c] = []

            for i in range(self.vocab_size):
                word_count = self.class_word_counts[c][i]

                prob = (word_count + 1) / (self.class_totals[c] + self.vocab_size)

                self.word_probs[c].append(prob)

    def predict(self, vector):
        """
        Predice la clase de un vector.
        """

        best_class = None
        best_log_prob = float('-inf')

        for c in self.classes:
            # log P(c)
            log_prob = math.log(self.class_probs[c])

            # sumar log P(w|c)
            for i, count in enumerate(vector):
                if count > 0:
                    log_prob += count * math.log(self.word_probs[c][i])

            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_class = c

        return best_class

    def predict_all(self, X):
        return [self.predict(x) for x in X]