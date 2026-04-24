import os
import pandas as pd
from preprocessing.text_cleaning import clean_text


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

    # 4. Mostrar información útil
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