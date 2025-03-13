import mlflow.sklearn

# Definimos la ruta local donde guardamos el modelo con mlflow.sklearn.save_model
MODEL_PATH = "model_export"

# Cargamos el modelo al iniciar la aplicación (un pipeline de scikit-learn)
model = mlflow.sklearn.load_model(MODEL_PATH)

# Pruebas locales del servidor para verificar correcta inferencia
pos_example = "I loved this movie, it was amazing!"
neg_example = "I hated this movie, it was terrible!"

print(f'Ejemplo positivo: {model.predict([pos_example])[0]}')
print(f'Ejemplo negativo: {model.predict([neg_example])[0]}')


