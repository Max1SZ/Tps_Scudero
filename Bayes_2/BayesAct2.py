import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("SISTEMA DE DETECCIÓN DE GROOMING")
print("Clasificador Naive Bayes para Mensajes")
print("="*70)

# === CREAR DATASET DE MENSAJES ===
print("\n" + "="*70)
print("PASO 1: CARGA DE DATOS")
print("="*70)

# Dataset de mensajes etiquetados
datos = {
    'mensaje': [
        '¿Cuántos años tenés? No le digas a nadie que hablamos.',
        'Hola, ¿cómo estás? ¿Querés jugar algo online?',
        'Sos muy linda, no le muestres esto a tus papás.',
        '¿Terminaste la tarea de matemáticas?',
        'Podés mandarme una foto tuya, pero que sea solo para mí.',
        '¿Querés que te ayude con el juego que no podías pasar?',
        'No le contés a nadie que hablamos por acá, ¿sí?',
        '¿Te gustó la película de ayer?',
        '¿Estás solo en casa ahora?',
        '¿Querés hacer una videollamada para estudiar juntos?',
        'No le digas a tus padres que te escribí, es nuestro secreto.',
        '¿Cómo te fue en el examen de historia?',
        'Mandame una foto tuya, pero que nadie más la vea.',
        '¿Jugamos Minecraft esta tarde?',
        'Sos muy especial para mí, no le cuentes a nadie lo que hablamos.',
        '¿Querés que te pase los apuntes de biología?',
        '¿Podés mostrarme cómo estás vestida ahora?',
        '¿Tenés ganas de salir a andar en bici mañana?',
        'No hace falta que le digas a nadie que hablamos tanto.',
        '¿Querés que estudiemos juntos para el parcial?'
    ],
    'etiqueta': [
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming',
        'grooming',
        'no grooming'
    ]
}

df = pd.DataFrame(datos)

print(f"\n✓ Dataset cargado exitosamente")
print(f"Total de mensajes: {len(df)}")
print(f"\nDistribución de etiquetas:")
print(df['etiqueta'].value_counts())
print(f"\nPorcentaje de grooming: {(df['etiqueta']=='grooming').sum()/len(df)*100:.1f}%")

print("\n" + "="*70)
print("EJEMPLOS DE MENSAJES")
print("="*70)
print("\n Mensajes clasificados como GROOMING:")
for idx, row in df[df['etiqueta']=='grooming'].head(3).iterrows():
    print(f"\n  {idx+1}. \"{row['mensaje']}\"")

print("\n Mensajes clasificados como NO GROOMING:")
for idx, row in df[df['etiqueta']=='no grooming'].head(3).iterrows():
    print(f"\n  {idx+1}. \"{row['mensaje']}\"")

# === SEPARAR CARACTERÍSTICAS Y ETIQUETAS ===
print("\n" + "="*70)
print("PASO 2: PREPARACIÓN DE DATOS")
print("="*70)

X = df['mensaje']  # Características (texto de los mensajes)
y = df['etiqueta']  # Variable objetivo (grooming / no grooming)

print(f"\nCaracterísticas (X): {len(X)} mensajes de texto")
print(f"Etiquetas (y): {len(y)} clasificaciones")

# === VECTORIZACIÓN DEL TEXTO ===
print("\n" + "="*70)
print("PASO 3: VECTORIZACIÓN DEL TEXTO")
print("="*70)

# CountVectorizer convierte texto en números (conteo de palabras)
# Según el manual (pág 15-16), MultinomialNB trabaja con frecuencias de palabras
vectorizer = CountVectorizer()

print("\n |Convirtiendo mensajes de texto a vectores numéricos. . . .|")
print("   Método: Conteo de frecuencia de palabras (Bag of Words)")

# === DIVIDIR EN ENTRENAMIENTO Y PRUEBA ===
print("\n" + "="*70)
print("PASO 4: DIVISIÓN DE DATOS")
print("="*70)

# Dividir 70% entrenamiento, 30% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n Conjunto de entrenamiento: {len(X_train)} mensajes")
print(f" Conjunto de prueba: {len(X_test)} mensajes")

# Vectorizar por separado train y test
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print(f"\n Vocabulario aprendido: {len(vectorizer.get_feature_names_out())} palabras únicas")
print(f"\nEjemplos de palabras en el vocabulario:")
palabras_ejemplo = list(vectorizer.get_feature_names_out())[:10]
for palabra in palabras_ejemplo:
    print(f"   - {palabra}")

# === ENTRENAMOS EL CLASIFICADOR NAIVE BAYES ===
print("\n" + "="*70)
print("PASO 5: ENTRENAMIENTO DEL MODELO")
print("="*70)

# MultinomialNB: Para datos de conteo/frecuencia (texto)
# Según manual pág 15: "Se usa cuando las características son frecuencias o conteos"
modelo = MultinomialNB()

print("\n|Entrenando clasificador Naive Bayes. . . .|")
print("   Algoritmo: MultinomialNB")
print("   Base teórica: Teorema de Bayes (pág 8 del manual)")

modelo.fit(X_train_vectorized, y_train)
print("\n Modelo entrenado exitosamente")

# PASO 6: EVALUAR EL MODELO
print("\n" + "="*70)
print("PASO 6: EVALUACIÓN DEL MODELO")
print("="*70)

# Hacer predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test_vectorized)

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)

print(f"\n RESULTADOS DE LA EVALUACIÓN")
print(f"\nPrecisión del modelo (Accuracy): {accuracy*100:.2f}%")

# Reporte de clasificación detallado
print("\n" + "-"*70)
print("REPORTE DE CLASIFICACIÓN DETALLADO")
print("-"*70)
print("\nSegún manual pág 18, el reporte muestra:\n\
- Precision: De los que predije como grooming, ¿cuántos lo eran?\n\
- Recall: De todos los grooming reales, ¿cuántos detecté?\n\
- F1-score: Balance entre precision y recall\n")

print(classification_report(y_test, y_pred, 
                          target_names=['grooming', 'no grooming'],
                          zero_division=0))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred, labels=['grooming', 'no grooming'])

print("\n" + "-"*70)
print("MATRIZ DE CONFUSIÓN")
print("-"*70)
print("\nSegún manual pág 18-19:")
print("  TP (True Positive): Predijo grooming y era grooming ✓")
print("  TN (True Negative): Predijo no grooming y era no grooming ✓")
print("  FP (False Positive): Predijo grooming pero era no grooming ✗")
print("  FN (False Negative): Predijo no grooming pero era grooming ✗ (PELIGROSO)\n")

print(pd.DataFrame(cm, 
                   index=['Real: grooming', 'Real: no grooming'],
                   columns=['Pred: grooming', 'Pred: no grooming']))

# Análisis de errores críticos
fn_count = cm[0][1]  # Falsos Negativos
if fn_count > 0:
    print(f"\n ATENCIÓN: {fn_count} caso(s) de grooming NO detectado(s)")
    print("   Este es el error más peligroso: no detectar grooming real")

# === CLASIFICAR NUEVOS MENSAJES ===
print("\n" + "="*70)
print("PASO 7: CLASIFICACIÓN DE NUEVOS MENSAJES")
print("="*70)

nuevos_mensajes = [
    "¿Podés mandarme una foto tuya? No se la muestres a nadie.",
    "¿Jugamos Minecraft esta tarde?",
    "No le digas a tus papás que hablamos, ¿sí?",
    "¿Terminaste el trabajo práctico de historia?"
]

print("\n Analizando nuevos mensajes...\n")

# Vectorizar nuevos mensajes
nuevos_vectorizados = vectorizer.transform(nuevos_mensajes)

# Predecir
predicciones = modelo.predict(nuevos_vectorizados)

# Obtener probabilidades
probabilidades = modelo.predict_proba(nuevos_vectorizados)

# Mostrar resultados
for i, mensaje in enumerate(nuevos_mensajes):
    clasificacion = predicciones[i]
    prob_grooming = probabilidades[i][0] if clasificacion == 'grooming' else probabilidades[i][1]
    prob_no_grooming = probabilidades[i][1] if clasificacion == 'grooming' else probabilidades[i][0]
    
    alerta = "!!!" if clasificacion == "grooming" else "(OK)"
    
    print(f"{alerta} Mensaje {i+1}:")
    print(f'   "{mensaje}"')
    print(f"   Clasificación: {clasificacion.upper()}")
    print(f"   Probabilidad grooming: {prob_grooming*100:.1f}%")
    print(f"   Probabilidad no grooming: {prob_no_grooming*100:.1f}%")
    print()

# === VISUALIZACIONES ===
print("\n" + "="*70)
print("PASO 8: GENERACIÓN DE GRÁFICOS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
fig.suptitle('Sistema de Detección de Grooming - Análisis Completo', 
             fontsize=16, fontweight='bold')

# Gráfico 1: Matriz de Confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=axes[0,0],
            xticklabels=['Pred: grooming', 'Pred: no grooming'],
            yticklabels=['Real: grooming', 'Real: no grooming'],
            cbar_kws={'label': 'Cantidad'})
axes[0,0].set_title('Matriz de Confusión', fontsize=12, fontweight='bold')
axes[0,0].set_ylabel('Valor Real', fontsize=10)
axes[0,0].set_xlabel('Predicción del Modelo', fontsize=10)

# Gráfico 2: Distribución de etiquetas
etiquetas_count = df['etiqueta'].value_counts()
colors = ['#e74c3c', '#2ecc71']
axes[0,1].pie(etiquetas_count, labels=['Grooming', 'No Grooming'], 
              autopct='%1.1f%%', colors=colors, startangle=90)
axes[0,1].set_title('Distribución de Mensajes en Dataset', 
                    fontsize=12, fontweight='bold')

# Gráfico 3: Métricas del modelo
metricas = classification_report(y_test, y_pred, 
                                target_names=['grooming', 'no grooming'],
                                output_dict=True, zero_division=0)

categorias = ['Precision', 'Recall', 'F1-Score']
grooming_valores = [metricas['grooming']['precision'], 
                   metricas['grooming']['recall'],
                   metricas['grooming']['f1-score']]
no_grooming_valores = [metricas['no grooming']['precision'],
                      metricas['no grooming']['recall'],
                      metricas['no grooming']['f1-score']]

x = np.arange(len(categorias))
width = 0.35

axes[1,0].bar(x - width/2, grooming_valores, width, label='Grooming', color='#e74c3c')
axes[1,0].bar(x + width/2, no_grooming_valores, width, label='No Grooming', color='#2ecc71')
axes[1,0].set_ylabel('Valor', fontsize=10)
axes[1,0].set_title('Métricas por Clase', fontsize=12, fontweight='bold')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(categorias)
axes[1,0].legend()
axes[1,0].set_ylim(0, 1.1)

# Añadir valores sobre las barras
for i, v in enumerate(grooming_valores):
    axes[1,0].text(i - width/2, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)
for i, v in enumerate(no_grooming_valores):
    axes[1,0].text(i + width/2, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)

# Gráfico 4: Probabilidades de nuevos mensajes
labels = [f'Msg {i+1}' for i in range(len(nuevos_mensajes))]
prob_grooming_list = []
prob_no_grooming_list = []

for i in range(len(nuevos_mensajes)):
    if predicciones[i] == 'grooming':
        prob_grooming_list.append(probabilidades[i][0])
        prob_no_grooming_list.append(probabilidades[i][1])
    else:
        prob_grooming_list.append(probabilidades[i][1])
        prob_no_grooming_list.append(probabilidades[i][0])

x = np.arange(len(labels))
axes[1,1].bar(x - width/2, prob_grooming_list, width, label='P(Grooming)', color='#e74c3c')
axes[1,1].bar(x + width/2, prob_no_grooming_list, width, label='P(No Grooming)', color='#2ecc71')
axes[1,1].set_ylabel('Probabilidad', fontsize=10)
axes[1,1].set_title('Probabilidades de Nuevos Mensajes', fontsize=12, fontweight='bold')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(labels)
axes[1,1].legend()
axes[1,1].set_ylim(0, 1.1)

plt.tight_layout()
plt.show()

print("\n Gráficos generados exitosamente")

# PASO 9: ANÁLISIS DE PALABRAS CLAVE
print("\n" + "="*70)
print("PASO 9: ANÁLISIS DE PALABRAS CLAVE")
print("="*70)

# Obtener las probabilidades log de cada palabra para cada clase
feature_names = vectorizer.get_feature_names_out()
grooming_idx = list(modelo.classes_).index('grooming')
no_grooming_idx = list(modelo.classes_).index('no grooming')

# Log probabilidades de las palabras
log_prob_grooming = modelo.feature_log_prob_[grooming_idx]
log_prob_no_grooming = modelo.feature_log_prob_[no_grooming_idx]

# Diferencia (palabras más asociadas a grooming)
diferencia = log_prob_grooming - log_prob_no_grooming

# Top palabras asociadas a grooming
top_grooming_indices = diferencia.argsort()[-10:][::-1]
print("\n Palabras más asociadas a GROOMING:")
for idx in top_grooming_indices:
    print(f"   - '{feature_names[idx]}' (score: {diferencia[idx]:.3f})")

# Top palabras asociadas a no grooming
top_no_grooming_indices = diferencia.argsort()[:10]
print("\n✅ Palabras más asociadas a NO GROOMING:")
for idx in top_no_grooming_indices:
    print(f"   - '{feature_names[idx]}' (score: {diferencia[idx]:.3f})")

# PASO 10: INFORME FINAL
print("\n" + "="*70)
print("INFORME FINAL DEL SISTEMA")
print("="*70)

print(f"""
--------------------------------------------------
RESUMEN DEL MODELO
--------------------------------------------------
✓ Algoritmo utilizado: Naive Bayes (MultinomialNB)
✓ Base teórica: Teorema de Bayes (Manual pág 8-11)
✓ Precisión general: {accuracy*100:.2f}%
✓ Total de mensajes analizados: {len(df)}
✓ Vocabulario aprendido: {len(feature_names)} palabras

--------------------------------------
RENDIMIENTO POR CLASE
--------------------------------------
Clase GROOMING:
  • Precision: {metricas['grooming']['precision']:.2%}
  • Recall: {metricas['grooming']['recall']:.2%}
  • F1-Score: {metricas['grooming']['f1-score']:.2%}

Clase NO GROOMING:
  • Precision: {metricas['no grooming']['precision']:.2%}
  • Recall: {metricas['no grooming']['recall']:.2%}
  • F1-Score: {metricas['no grooming']['f1-score']:.2%}

----------------------------------
ANÁLISIS DE ERRORES
----------------------------------
""")

if fn_count > 0:
    print(f"CRÍTICO: {fn_count} caso(s) de grooming no detectado(s)")
    print("   → Estos son los errores más peligrosos")
    print("   → Recomendación: Revisar manualmente mensajes con baja confianza")
else:
    print("✓ No se detectaron falsos negativos en el conjunto de prueba")

fp_count = cm[1][0]
if fp_count > 0:
    print(f"\nMODERADO: {fp_count} falsa(s) alarma(s)")
    print("   → Mensajes inocentes clasificados como grooming")
    print("   → Impacto: Revisión manual adicional")
else:
    print("\n✓ No se detectaron falsos positivos en el conjunto de prueba")

print(f"""
-----------------------------------
RECOMENDACIONES PARA LA PLATAFORMA
---------------------------------------
1. IMPLEMENTACIÓN EN PRODUCCIÓN:
   • Revisar MANUALMENTE todos los mensajes marcados como grooming
   • No bloquear automáticamente sin revisión humana
   • Priorizar mensajes con alta probabilidad de grooming

2. MEJORA CONTINUA:
   • Reentrenar el modelo mensualmente con nuevos casos
   • Incorporar feedback de moderadores humanos
   • Expandir dataset con más ejemplos variados

3. SISTEMA DE ALERTAS:
   • Alerta ALTA: Probabilidad > 80%
   • Alerta MEDIA: Probabilidad 50-80%
   • Alerta BAJA: Probabilidad < 50%

4. PRIVACIDAD Y ÉTICA:
   • Cumplir con regulaciones de protección de menores
   • Logs de auditoría para revisión de casos
   • Protocolo de escalamiento a autoridades cuando corresponda

--------------------------------------------
ANÁLISIS COMPLETADO
--------------------------------------------
""")