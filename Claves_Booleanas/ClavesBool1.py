import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Descarga de recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Documentos sobre inteligencia artificial
documentos = {
    "doc1": "La inteligencia artificial está revolucionando la tecnologia.",
    "doc2": "El aprendizaje automático es clave en la inteligencia artificial.",
    "doc3": "Procesamiento del lenguaje natural y redes neuronales.",
    "doc4": "Las redes neuronales son fundamentales en deep learning.",
    "doc5": "El futuro de la IA está en el aprendizaje profundo."
}

# Función para limpiar y tokenizar texto
def limpiar_texto(texto):
    # Convierte a minúsculas, tokeniza y elimina signos de puntuación y stopwords
    tokens = word_tokenize(texto.lower())
    stop_words = set(stopwords.words('spanish') + list(string.punctuation))
    return [t for t in tokens if t not in stop_words]

# Crear índice invertido: relaciona cada palabra clave con los documentos donde aparece
indice = {}
for doc_id, texto in documentos.items():
    palabras = limpiar_texto(texto)
    for palabra in palabras:
        if palabra not in indice:
            indice[palabra] = set()
        indice[palabra].add(doc_id)

# Función que ejecuta la búsqueda booleana
def procesar_consulta(consulta):
    tokens = consulta.lower().split()
    if len(tokens) != 3:
        return set()
    
    op1, operador, op2 = tokens
    docs1 = indice.get(op1, set())
    docs2 = indice.get(op2, set())

    if operador == "and":
        return docs1 & docs2  # Intersección
    elif operador == "or":
        return docs1 | docs2  # Unión
    elif operador == "not":
        return docs1 - docs2  # Diferencia
    else:
        return set()

# Bucle principal que permite al usuario hacer búsquedas
while True:
    consulta = input("Ingrese una consulta booleana (o 'salir'): ")
    if consulta.lower() == "salir":
        break
    resultado = procesar_consulta(consulta)
    print(" Documentos encontrados:", resultado)
