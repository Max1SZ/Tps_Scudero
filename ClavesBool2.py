import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Descargar recursos (si no se hizo ya)
nltk.download('punkt')
nltk.download('stopwords')

# Documentos sobre civilizaciones antiguas
documentos = {
    "doc1": "Los egipcios construyeron las pirámides y desarrollaron una escritura jeroglífica.",
    "doc2": "La civilización romana fue una de las más influyentes en la historia occidental.",
    "doc3": "Los mayas eran expertos astrónomos y tenían un avanzado sistema de escritura.",
    "doc4": "La antigua Grecia sentó las bases de la democracia y la filosofía moderna.",
    "doc5": "Los sumerios inventaron la escritura cuneiforme y fundaron las primeras ciudades."
}

# Función para limpiar y tokenizar
def limpiar_texto(texto):
    tokens = word_tokenize(texto.lower())
    stop_words = set(stopwords.words('spanish') + list(string.punctuation))
    return [t for t in tokens if t not in stop_words]

# Crear índice invertido
indice = {}
for doc_id, texto in documentos.items():
    palabras = limpiar_texto(texto)
    for palabra in palabras:
        if palabra not in indice:
            indice[palabra] = set()
        indice[palabra].add(doc_id)

# Función de búsqueda booleana
def procesar_consulta(consulta):
    tokens = consulta.lower().split()
    if len(tokens) != 3:
        return set()
    
    op1, operador, op2 = tokens
    docs1 = indice.get(op1, set())
    docs2 = indice.get(op2, set())

    if operador == "and":
        return docs1 & docs2
    elif operador == "or":
        return docs1 | docs2
    elif operador == "not":
        return docs1 - docs2
    else:
        return set()

# Interfaz de usuario
while True:
    consulta = input("Ingrese una consulta booleana (o 'salir'): ")
    if consulta.lower() == "salir":
        break
    resultado = procesar_consulta(consulta)
    print("Documentos encontrados:", resultado)
