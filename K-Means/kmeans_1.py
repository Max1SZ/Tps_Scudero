import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1️⃣ Cargar datos
# ==========================================
data = {
    "Jurisdicción": [
        "Ciudad Autónoma de Buenos Aires",
        "Buenos Aires",
        "Catamarca",
        "Chaco",
        "Chubut",
        "Córdoba",
        "Corrientes",
        "Entre Ríos",
        "Formosa",
        "Jujuy",
        "La Pampa",
        "La Rioja",
        "Mendoza",
        "Misiones",
        "Neuquén",
        "Río Negro",
        "Salta",
        "San Juan",
        "San Luis",
        "Santa Cruz",
        "Santa Fe",
        "Santiago del Estero",
        "Tierra del Fuego, Antártida e Islas del Atlántico Sur",
        "Tucumán"
    ],
    "Viviendas Habitadas": [
        1391258,
        5970702,
        131978,
        368728,
        213317,
        1378237,
        370958,
        494473,
        194689,
        238141,
        140879,
        124149,
        639467,
        420101,
        254545,
        276371,
        404504,
        241436,
        182886,
        118047,
        1273460,
        311361,
        65535,
        493794
    ]
}

df = pd.DataFrame(data)

# ==========================================
# 2️⃣ Escalar datos
# ==========================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['Viviendas Habitadas']])

# ==========================================
# 3️⃣ Calcular inercia para distintos K (gráfico del codo)
# ==========================================
inercia = []
K = range(1, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K, inercia, marker='o', color='b')
plt.title('Método del Codo (Elbow Method)')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.xticks(K)
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# 4️⃣ Aplicar K-Means con k=3
# ==========================================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Centroides en escala original
centroides_escalados = kmeans.cluster_centers_.flatten().reshape(-1, 1)
centroides_original = scaler.inverse_transform(centroides_escalados).flatten()

# Etiquetas interpretables
orden_clusters = np.argsort(centroides_original)
map_labels = {
    orden_clusters[0]: "Pocas viviendas",
    orden_clusters[1]: "Cantidad normal de viviendas",
    orden_clusters[2]: "Muchas viviendas"
}
df['Segmento'] = df['cluster'].map(map_labels)

# ==========================================
# 5️⃣ Mostrar resultados
# ==========================================
print("\nCentroides (viviendas) por cluster:")
for i, c in enumerate(centroides_original):
    print(f"  Cluster {i} -> {map_labels[i]}: {int(round(c))} viviendas")

print("\nAsignación de segmentos:")
print(df[['Jurisdicción', 'Viviendas Habitadas', 'Segmento']].sort_values('Viviendas Habitadas', ascending=False).to_string(index=False))

# ==========================================
# 6️⃣ Visualización de clusters
# ==========================================
plt.figure(figsize=(12,6))
x = np.arange(len(df))
for cluster_id in sorted(df['cluster'].unique()):
    grupo = df[df['cluster'] == cluster_id]
    plt.scatter(grupo.index, grupo['Viviendas Habitadas'], s=100, label=map_labels[cluster_id])

for cluster_id in range(len(centroides_original)):
    plt.hlines(centroides_original[cluster_id], xmin=-0.5, xmax=len(df)-0.5, linestyles='dashed')

plt.xticks(x, df['Jurisdicción'], rotation=90)
plt.xlabel('Jurisdicción')
plt.ylabel('Viviendas Habitadas')
plt.title('Agrupamiento de jurisdicciones por cantidad de viviendas (K-Means, 3 clusters)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
