import os
import pandas as pd

def leer_archivos_en_directorio(directorio, sentimiento, tipo):
    """
    Lee todos los archivos de texto en 'directorio' y devuelve una lista de diccionarios,
    cada uno con el texto, el sentimiento (pos/neg) y si es train o test.
    """
    registros = []
    for filename in os.listdir(directorio):
        if filename.endswith(".txt"):
            ruta_archivo = os.path.join(directorio, filename)
            with open(ruta_archivo, "r", encoding="utf-8") as f:
                texto = f.read()
            registros.append({
                "review": texto,
                "sentiment": sentimiento,  # 1 = positivo, 0 = negativo
                #"split": tipo              # 'train' o 'test'
            })
    return registros

# Rutas (ajusta según tu sistema)
ruta_train_pos = "data/train/pos"
ruta_train_neg = "data/train/neg"
ruta_test_pos  = "data/test/pos"
ruta_test_neg  = "data/test/neg"

# Leemos los archivos de cada carpeta
train_pos = leer_archivos_en_directorio(ruta_train_pos, sentimiento=1, tipo="train")
train_neg = leer_archivos_en_directorio(ruta_train_neg, sentimiento=0, tipo="train")
test_pos  = leer_archivos_en_directorio(ruta_test_pos,  sentimiento=1, tipo="test")
test_neg  = leer_archivos_en_directorio(ruta_test_neg,  sentimiento=0, tipo="test")

# Convertimos las listas de datos de train y test a un DataFrame de pandas
train_df = pd.DataFrame(train_pos + train_neg)
test_df = pd.DataFrame(test_pos + test_neg)

# Desordenamos aleatoriamente los DataFrames
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Verificamos un poco la estructura de la data de entrenamiento
print("Verificando estructura de data de entrenamiento")
print(train_df.head())
print(train_df.tail())
print("Total de reseñas de entrenamiento:", len(train_df)) 

# Exportamos a CSV (sin índice)
train_df.to_csv("data/imdb_reviews_train.csv", index=False, encoding="utf-8")

# Verificamos un poco la estructura de la data de prueba
print("Verificando estructura de data de prueba")
print(test_df.head())
print(test_df.tail())
print("Total de reseñas de prueba:", len(test_df)) 

# Exportamos a CSV (sin índice)
test_df.to_csv("data/imdb_reviews_test.csv", index=False, encoding="utf-8")
