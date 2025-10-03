import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split #dividir datos en cojunto de entrenamiento y otro de prueba
from sklearn.metrics import accuracy_score, confusion_matrix 

data = {
    'texto': [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"
    ],
    'etiqueta': [
        'real', 'fake', 'real', 'fake', 'real',
        'fake', 'real', 'fake', 'real', 'fake'
    ]
}
df = pd.DataFrame(data)
#separamos caracteristicas y etiquetas
X = df['texto']
y = df['etiqueta']

#convertir texto a una matriz de conteo de palabras
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

#dividir datos en conjunto de entrenamiento y otro de prueba
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, 
                                y, test_size=0.2, random_state=42)
#random state es para que la division sea la misma

#crear y entrenar el modelo Naive Bayes
model = MultinomialNB()

#entrenar el modelo
model.fit(X_train, y_train)

#hacer predicciones
y_pred = model.predict(X_test)

#evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)

#matriz de confusion
conf_matrix = confusion_matrix(y_test, y_pred, labels=['real', 'fake'])

#nuevo texto a clasificar
nuevas_noticias = [
    "Nuevo estudio demuestra que el cafe mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]
#transformar el nuevo texto usando el mismo vectorizador
nuevas_noticias_vect = vectorizer.transform(nuevas_noticias)

#predecir las etiquetas para el nuevo texto
predicciones_nuevas = model.predict(nuevas_noticias_vect)


#mostrar resultados
print("Matriz de confusión:")
print(f"Precisión del modelo: {accuracy:.2f}")
print("matriz de confusion:")
print(conf_matrix)
print("\nPredicciones para nuevas noticias:")
for noticia, etiqueta in zip(nuevas_noticias, predicciones_nuevas):
    print(f"Noticia: '{noticia}' → Predicción: {etiqueta}")



