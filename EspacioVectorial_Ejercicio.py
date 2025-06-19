# Importar las librerías necesarias
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Documentos de ejemplo
documents = [
    "El veloz zorro marrón salta sobre el perro perezoso.",
    "Un perro marrón persiguió al zorro.",
    "El perro es perezoso."
]

# Convertir documentos a vectores TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Calcular la similitud del coseno entre los documentos
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Graficar la matriz de similitud como un mapa de calor
plt.figure(figsize=(8, 6))  # Tamaño del gráfico
sns.heatmap(cosine_sim, annot=True, cmap="Reds", xticklabels=[f"Doc{i+1}" for i in range(len(documents))],
yticklabels=[f"Doc{i+1}" for i in range(len(documents))]) # Mostrar los valores en cada celda, Paleta de colores
plt.title("Matriz de Similitud del Coseno")
plt.show()