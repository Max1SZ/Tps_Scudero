import math 
import pandas as pd

def entropia(p, n):
    """Calcula la entropía de un conjunto con
      p positivos y n negativos"""
    if p == 0 or n == 0:
        return 0
    total = p + n
    p_ratio = p / total
    n_ratio = n / total
    return - (p_ratio * math.log2(p_ratio) + n_ratio * math.log2(n_ratio))

def ganancia_info(total_p, total_n, divisiones):
    """
    Calcula la ganancia de información
    divisiones: lista de tuplas (positivos, negativos)
    """
    entropia_total = entropia(total_p, total_n)
    total = total_p + total_n
    entropia_ponderada = 0
    for (p, n) in divisiones:
        entropia_ponderada += ((p+n)/total) * entropia(p, n)
    return entropia_total, entropia_ponderada, entropia_total - entropia_ponderada

# ============================
# 1. Cargar los datos
# ============================
data = pd.DataFrame({
    "Edad": [24, 38, 29, 45, 52, 33, 41, 27, 36, 31],
    "UsoGB": [2.5, 6.0, 3.0, 8.0, 7.5, 4.0, 5.5, 2.0, 6.5, 3.5],
    "LineaFija": ["No", "Sí", "No", "Sí", "Sí", "No", "Sí", "No", "Sí", "No"],
    "Acepta": ["No", "Sí", "No", "Sí", "Sí", "No", "Sí", "No", "Sí", "No"]
})
print("=== Conjunto de datos ===")
print(data)
# ============================
# 2. Entropía del conjunto original
# ============================
p_total = sum(data["Acepta"] == "Sí")
n_total = sum(data["Acepta"] == "No")
print("\n=== 1. ENTROPÍA DEL CONJUNTO ORIGINAL ===")
print(f"Positivos (Sí): {p_total}, Negativos (No): {n_total}")
print(f"Entropía total: {entropia(p_total, n_total)}")
# ============================
# 3. Evaluar atributos
# ============================
print("\n=== 2. GANANCIA DE INFORMACIÓN POR ATRIBUTO ===")
#Edad va a ser agrupada en rangos: joven <= 30, audlto 31 hasta 50 y mayor >= 50
#sacamos tambien ganancia de Tiene linea fija
#uso de datos sera agrupado en: bajo <=3gb, medio 3.1-6gb, alto >6gb
# ---- Edad agrupada (clasifica las categorias)----
bins_edad = [0, 30, 50, 100]
labels_edad = ["Joven", "Adulto", "Mayor"]
data["EdadGrupo"] = pd.cut(data["Edad"], bins=bins_edad, labels=labels_edad, right=True)

#esto crea una tabla de contingencia para la edad agrupada
#la funcion observed=False es para evitar un warning de pandas
tabla_edad = data.groupby("EdadGrupo", observed=False)["Acepta"].value_counts().unstack().fillna(0)
print("\n-- Edad agrupada --")
print(tabla_edad)
# calculamos ganancia de informacion a continuacion
entropia_total, entropia_ponderada, ganancia = ganancia_info(
    p_total, n_total,
    [(row.get("Sí", 0), row.get("No", 0)) for idx, row in tabla_edad.iterrows()]
)
print(f"Edad → Ganancia: {ganancia:.4f}")
# ---- Línea fija ----
#repetimos lo mismo que con edad pero sin la agrupacion.
tabla_linea = data.groupby("LineaFija", observed=False)["Acepta"].value_counts().unstack().fillna(0)
print("\n-- Línea fija --") 
print(tabla_linea)
entropia_total, entropia_ponderada, ganancia = ganancia_info(
    p_total, n_total,
    [(row.get("Sí", 0), row.get("No", 0)) for idx, row in tabla_linea.iterrows()]
)
print(f"Línea fija → Ganancia: {ganancia}")
# ---- Uso de datos agrupado ----   
bins_uso = [0, 3, 6, 100]
labels_uso = ["Bajo", "Medio", "Alto"]
data["UsoGrupo"] = pd.cut(data["UsoGB"], bins=bins_uso, labels=labels_uso, right=True)
tabla_uso = data.groupby("UsoGrupo", observed=False)["Acepta"].value_counts().unstack().fillna(0)
print("\n-- Uso de datos agrupado --")
print(tabla_uso)
entropia_total, entropia_ponderada, ganancia = ganancia_info(
    p_total, n_total,
    [(row.get("Sí", 0), row.get("No", 0)) for idx, row in tabla_uso.iterrows()]
)
print(f"Uso de datos → Ganancia: {ganancia}")