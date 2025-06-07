import chardet #librería que detecta la codificación de archivos, 
               #asi entiende los caracteres especiales
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import pandas as pd

#Pedir al usuario la ruta del archivo
ruta = ("c:/Users/estudiante/Desktop/TareasPYTHON/2025/CorpusEducacion.txt")

#Detectar la codificación con chardet
with open(ruta, "rb") as archivo_binario:
    contenido_binario = archivo_binario.read()

resultado = chardet.detect(contenido_binario)
codificacion = resultado["encoding"]
print(f"Codificación detectada: {codificacion}")

#Leer el texto con la codificación correcta
with open(ruta, "r", encoding=codificacion) as archivo_texto:
    texto = archivo_texto.read()

#Dividir el texto en oraciones
oraciones = [linea.strip() for linea in texto.split('.') if linea.strip()]

#Crear el vectorizador de N-gramas
vectorizador = CountVectorizer(ngram_range=(2, 3), min_df=2)  # bigramas y trigramas, mínimo 2 repeticiones
X = vectorizador.fit_transform(oraciones)

#Obtener los N-gramas y sus frecuencias
ngrams = vectorizador.get_feature_names_out()
frecuencias = X.toarray().sum(axis=0)

#Crear un DataFrame para ordenar
df = pd.DataFrame({"ngram": ngrams, "frecuencia": frecuencias})
df["longitud"] = df["ngram"].apply(lambda x: len(x.split()))
df = df.sort_values(by="frecuencia", ascending=False)

#Separar bigramas y trigramas
bigrama = df[df["longitud"] == 2].head(10)
trigrama = df[df["longitud"] == 3].head(10)

#Graficar
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.barh(bigrama["ngram"], bigrama["frecuencia"], color='skyblue')
plt.title("Top 10 Bigrama")
plt.xlabel("Frecuencia")
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
plt.barh(trigrama["ngram"], trigrama["frecuencia"], color='salmon')
plt.title("Top 10 Trigrama")
plt.xlabel("Frecuencia")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
