import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === CARGAR DATOS DESDE EXCEL ===
print("="*70)
print("CARGA DE DATOS")
print("="*70)

ruta_archivo = input("\nIngresa la ruta del archivo Excel (o presiona Enter para usar datos de ejemplo): ").strip()
#C:/Users/estudiante/Desktop/TareasPYTHON/2025/TablaPrediccionAbandono-Entrenamiento.xlsx
if ruta_archivo == "":
    print("\n No se proporcionó archivo. Generando datos de ejemplo...")
    np.random.seed(42)
    n_estudiantes = 100
    
    datos = {
        'edad': np.random.randint(18, 35, n_estudiantes),
        'genero': np.random.choice(['M', 'F'], n_estudiantes),
        'carrera': np.random.choice(['Ciencia de Datos', 'Desarrollo Software', 'Redes'], n_estudiantes),
        'promedio_1er_cuatri': np.round(np.random.uniform(4.0, 10.0, n_estudiantes), 2),
        'materias_aprobadas': np.random.randint(0, 8, n_estudiantes),
        'materias_desaprobadas': np.random.randint(0, 5, n_estudiantes),
        'asistencia_promedio': np.round(np.random.uniform(40, 100, n_estudiantes), 1),
        'trabaja': np.random.choice(['Sí', 'No'], n_estudiantes, p=[0.6, 0.4]),
        'distancia_km': np.random.randint(1, 50, n_estudiantes),
        'participa_tutorias': np.random.choice(['Sí', 'No'], n_estudiantes, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(datos)
    
    def calcular_abandono(row):
        riesgo = 0
        if row['promedio_1er_cuatri'] < 6: riesgo += 2
        if row['materias_aprobadas'] < 4: riesgo += 2
        if row['materias_desaprobadas'] > 2: riesgo += 2
        if row['asistencia_promedio'] < 70: riesgo += 2
        if row['trabaja'] == 'Sí' and row['distancia_km'] > 30: riesgo += 1
        if row['participa_tutorias'] == 'No': riesgo += 1
        return 'abandonó' if riesgo >= 5 else 'continúa'
    
    df['estado_final'] = df.apply(calcular_abandono, axis=1)
    print("Datos de ejemplo generados exitosamente")
else:
    try:
        df = pd.read_excel(ruta_archivo)
        print(f"Archivo cargado exitosamente: {ruta_archivo}")
        
        mapeo_columnas = {
            'PromedioPrimerCuatrimestre': 'promedio_1er_cuatri',
            'CantMateriasAprobadasPrimerCuatrimestre': 'materias_aprobadas',
            'CantMateriasDesaprobadasPrimerCuatrimestre': 'materias_desaprobadas',
            'AsistenciaPromedio(%)': 'asistencia_promedio',
            'trabaja/NoTrabaja': 'trabaja',
            'DistanciaDomicilioAlInstituto(Kms)': 'distancia_km',
            'ActividadesExtracurriculares(Estudio)': 'participa_tutorias',
            'EstadoFinal': 'estado_final'
        }
        
        df.rename(columns=mapeo_columnas, inplace=True)
        print("Columnas renombradas correctamente")
        
    except Exception as e:
        print(f"\n Error al cargar el archivo: {e}")
        exit()

# === BLOQUE DE NORMALIZACIÓN Y MAPEO ===
print("\n" + "="*70)
print("NORMALIZACIÓN Y MAPEO DE VARIABLES")
print("="*70)

print("\nNormalizando valores de texto...")

if 'trabaja' in df.columns:
    df['trabaja'] = df['trabaja'].astype(str).str.strip().str.lower().replace({
        'si': 'sí', 's': 'sí', 'no': 'no'
    })

if 'participa_tutorias' in df.columns:
    df['participa_tutorias'] = df['participa_tutorias'].astype(str).str.strip().str.lower().replace({
        'si': 'sí', 's': 'sí', 'no': 'no'
    })

if 'estado_final' in df.columns:
    df['estado_final'] = df['estado_final'].astype(str).str.strip().str.lower().replace({
        'abandono': 'abandonó',
        'abandonó': 'abandonó',
        'continua': 'continúa',
        'continúa': 'continúa'
    })

print("Datos de texto normalizados correctamente.")

df_modelo = df.copy()

if 'genero' in df_modelo.columns:
    df_modelo['genero'] = df_modelo['genero'].str.strip().str.upper().map({'M': 0, 'F': 1})

carreras_unicas = df_modelo['carrera'].unique()
mapeo_carreras = {c: i for i, c in enumerate(sorted(carreras_unicas))}
df_modelo['carrera'] = df_modelo['carrera'].map(mapeo_carreras)

df_modelo['trabaja'] = df_modelo['trabaja'].map({'no': 0, 'sí': 1})
df_modelo['participa_tutorias'] = df_modelo['participa_tutorias'].map({'no': 0, 'sí': 1})
df_modelo['estado_final'] = df_modelo['estado_final'].map({'continúa': 0, 'abandonó': 1})

print("\nMapeos aplicados:")
print("- genero: M=0, F=1")
print(f"- carrera: {mapeo_carreras}")
print("- trabaja: no=0, sí=1")
print("- participa_tutorias: no=0, sí=1")
print("- estado_final: continúa=0, abandonó=1")

print("\nVerificando clases después del mapeo...")
print(df_modelo['estado_final'].value_counts(dropna=False))
print("Valores únicos:", df_modelo['estado_final'].unique())

if len(df_modelo['estado_final'].dropna().unique()) < 2:
    print("\n Error: solo se detectó una clase en 'estado_final'. Revisa los valores originales.")
    print("Ejemplo de valores originales:", df['estado_final'].unique())
    exit()

X = df_modelo.drop('estado_final', axis=1)
y = df_modelo['estado_final']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

modelo = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5, random_state=42)
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy*100:.2f}%")
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=['Continúa', 'Abandonó'], zero_division=0))

cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:")
print(cm)

importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': modelo.feature_importances_
}).sort_values('Importancia', ascending=False)

print("\nVariables más importantes para predecir el abandono:")
for idx, row in importancias.iterrows():
    if row['Importancia'] > 0:
        print(f"  {row['Variable']}: {row['Importancia']:.4f}")

# === VISUALIZACIONES ===
plt.figure(figsize=(12, 10))

# 1️⃣ Matriz de confusión
plt.subplot(2, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Continúa', 'Abandonó'],
            yticklabels=['Continúa', 'Abandonó'])
plt.title('Matriz de Confusión', fontsize=13, fontweight='bold')

# 2️⃣ Importancia de variables
plt.subplot(2, 2, 2)
importancias_top = importancias[importancias['Importancia'] > 0]
plt.barh(importancias_top['Variable'], importancias_top['Importancia'], color='steelblue')
plt.title('Importancia de Variables', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()

# 3️⃣ Distribución de clases
plt.subplot(2, 2, 3)
estado_counts = df['estado_final'].value_counts()
plt.pie(estado_counts, labels=estado_counts.index, autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c'], startangle=90)
plt.title('Distribución de Estudiantes', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# === Árbol de decisión en figura aparte ===
plt.figure(figsize=(15, 5))
plot_tree(modelo,
          feature_names=X.columns,
          class_names=['Continúa', 'Abandonó'],
          filled=True, rounded=True,
          fontsize=9)
plt.title('Árbol de Decisión', fontsize=16, fontweight='bold')
plt.show()

# === RECOMENDACIONES ===
print("\n" + "="*70)
print("RECOMENDACIONES PARA LA INSTITUCIÓN")
print("="*70)

print("\nBasándose en el análisis del modelo:")
print("\n1. FACTORES DE RIESGO IDENTIFICADOS:")
top_3_variables = importancias.head(3)
for idx, row in top_3_variables.iterrows():
    if row['Importancia'] > 0:
        print(f"   • {row['Variable']} (importancia: {row['Importancia']:.2%})")

print("\n2. ESTRATEGIAS SUGERIDAS:\n\
• Implementar alertas tempranas para estudiantes con:\n\
- Promedio del primer cuatrimestre < 6\n\
- Asistencia promedio < 70%\n\
- Más de 2 materias desaprobadas\n\
• Reforzar programa de tutorías (especialmente para quienes no participan)\n\
• Ofrecer flexibilidad horaria para estudiantes que trabajan\n\
• Programa de seguimiento personalizado en el primer año")

print("\n3. GRUPOS PRIORITARIOS:\n\
• Estudiantes que trabajan y viven lejos del instituto\n\
• Estudiantes con bajo rendimiento en el primer cuatrimestre\n\
• Estudiantes que no participan en actividades extracurriculares")

print("\n" + "="*70)
print("ANÁLISIS COMPLETADO")
print("="*70)