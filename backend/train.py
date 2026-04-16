import pandas as pd 
from preprocessing.text_cleaning import clean_text  

# 1. Cargar dataset desde archivo CSV
df = pd.read_csv("backend/data/dataset_og/customer_support_tickets.csv")

# 2. Crear nueva columna "text"
# Une el subject + description (mejor contexto para el modelo)
df["text"] = df["Ticket Subject"] + " " + df["Ticket Description"]

# 3. Quedarte solo con lo importante:
# - texto (input)
# - tipo de ticket (output)
df = df[["text", "Ticket Type"]]

# 4. Eliminar filas con valores nulos
# Evita errores al procesar texto
df = df.dropna()

# 5. Aplicar preprocesamiento a cada fila
# .apply() ejecuta la función clean_text en cada texto
df["tokens"] = df["text"].apply(clean_text)

# 6. Mostrar primeras filas para verificar resultados
print(df.head())