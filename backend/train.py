import os
import pandas as pd
from preprocessing.text_cleaning import clean_text
from model.bag_of_words import build_vocabulary, vectorize_dataset
from model.naive_bayes import NaiveBayes
from evaluation.kfold import k_fold_split
from evaluation.metrics import accuracy, precision_recall_f1, macro_f1, confusion_matrix

# Ruta absoluta del backend
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ruta del dataset nuevo
DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset_og",
    "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)


def load_dataset():
    """
    Carga el dataset Bitext desde CSV.
    """
    df = pd.read_csv(DATASET_PATH)
    return df


def prepare_data(df):
    """
    Prepara el dataset para clasificación.

    Entrada:
    - instruction: texto escrito por el usuario

    Salida:
    - category: clase que el modelo debe predecir
    """

    # Verificar que las columnas existan
    required_columns = ["instruction", "category"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatoria: {col}")

    # Crear columnas estándar para nuestro proyecto
    df["text"] = df["instruction"]
    df["label"] = df["category"]

    # Quedarnos solo con lo necesario
    df = df[["text", "label"]]

    # Eliminar filas vacías
    df = df.dropna()

    return df


def main():
    # 1. Cargar dataset
    df = load_dataset()

    # 2. Preparar columnas
    df = prepare_data(df)

    # 3. Aplicar limpieza
    df["tokens"] = df["text"].apply(clean_text)

    # 4. Construir vocabulario (BoW)
    vocab = build_vocabulary(df["tokens"])

    print("\nTamaño del vocabulario:")
    print(len(vocab))

    # Vectorizar dataset
    X = vectorize_dataset(df["tokens"], vocab)

    print("\nEjemplo de vector:")
    print(X[0][:20])  # primeros 20 valores

    #5. Naive Bayes Labels
    y = df["label"].tolist()

    # Crear modelo
    model = NaiveBayes()

    # Entrenar
    model.train(X, y)

    print("\nModelo entrenado!")

    # Probar con un ejemplo
    pred = model.predict(X[0])

    print("\nPredicción ejemplo:")
    print(pred)
    print("Real:", y[0])


    #6. métricas de validacion
    folds = k_fold_split(X, y, k=5)

    accuracies = []
    macro_f1_scores = []

    for i, (X_train, y_train, X_test, y_test) in enumerate(folds):

        model = NaiveBayes()
        model.train(X_train, y_train)

        y_pred = model.predict_all(X_test)

        acc = accuracy(y_test, y_pred)
        prec, rec, f1_per_class = precision_recall_f1(y_test, y_pred)
        mf1 = macro_f1(f1_per_class)

        accuracies.append(acc)
        macro_f1_scores.append(mf1)

        print(f"\nFold {i+1}")
        print("Accuracy:", acc)
        print("Macro F1:", mf1)

        cm = confusion_matrix(y_test, y_pred)

        print("\nMatriz de confusión:")
        for real_class in cm:
            print(real_class, cm[real_class])

    print("\nResultados Finales:")
    print("Accuracy promedio:", sum(accuracies) / len(accuracies))
    print("Macro F1 promedio:", sum(macro_f1_scores) / len(macro_f1_scores))


    # 6. Mostrar información útil
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