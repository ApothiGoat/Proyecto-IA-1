import os
import pickle
import pandas as pd

from preprocessing.text_cleaning import clean_text
from model.bag_of_words import build_vocabulary, vectorize_dataset
from model.naive_bayes import NaiveBayes
from evaluation.kfold import k_fold_split
from evaluation.metrics import accuracy, precision_recall_f1, macro_f1, confusion_matrix


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset_og",
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_dataset():
    df = pd.read_csv(DATASET_PATH)
    return df


def prepare_data(df):
    required_columns = ["instruction", "category"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")

    df["text"] = df["instruction"]
    df["label"] = df["category"]

    df = df[["text", "label"]]
    df = df.dropna()

    return df


def save_model(model, vocab):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "naive_bayes.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(MODEL_DIR, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    print("\nModelo y vocabulario guardados correctamente.")


def main():
    df = load_dataset()
    df = prepare_data(df)

    df["tokens"] = df["text"].apply(clean_text)

    vocab = build_vocabulary(df["tokens"])

    print("\nTamaño del vocabulario:")
    print(len(vocab))

    X = vectorize_dataset(df["tokens"], vocab)
    y = df["label"].tolist()

    print("\nEjemplo de vector:")
    print(X[0][:20])

    # Entrenamiento de prueba
    model = NaiveBayes()
    model.train(X, y)

    print("\nModelo entrenado!")

    pred = model.predict(X[0])

    print("\nPredicción ejemplo:")
    print(pred)
    print("Real:", y[0])

    # K-Folds
    folds = k_fold_split(X, y, k=5)

    accuracies = []
    macro_f1_scores = []

    for i, (X_train, y_train, X_test, y_test) in enumerate(folds):
        fold_model = NaiveBayes()
        fold_model.train(X_train, y_train)

        y_pred = fold_model.predict_all(X_test)

        acc = accuracy(y_test, y_pred)
        prec, rec, f1_per_class = precision_recall_f1(y_test, y_pred)
        mf1 = macro_f1(f1_per_class)

        accuracies.append(acc)
        macro_f1_scores.append(mf1)

        print(f"\nFold {i + 1}")
        print("Accuracy:", acc)
        print("Macro F1:", mf1)

        cm = confusion_matrix(y_test, y_pred)

        print("\nMatriz de confusión:")
        for real_class in cm:
            print(real_class, cm[real_class])

    print("\nResultados Finales:")
    print("Accuracy promedio:", sum(accuracies) / len(accuracies))
    print("Macro F1 promedio:", sum(macro_f1_scores) / len(macro_f1_scores))

    # Modelo final entrenado con todo el dataset
    final_model = NaiveBayes()
    final_model.train(X, y)

    save_model(final_model, vocab)

    print("\nPrimeras filas procesadas:")
    print(df.head())

    print("\nCantidad de registros:")
    print(len(df))

    print("\nClases encontradas:")
    print(df["label"].value_counts())

    print("\nEjemplo de texto original:")
    print(df.iloc[0]["text"])

    print("\nEjemplo de tokens:")
    print(df.iloc[0]["tokens"])


if __name__ == "__main__":
    main()