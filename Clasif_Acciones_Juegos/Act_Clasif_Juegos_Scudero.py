import pandas as pd

# Crear el DataFrame con los datos de ejemplo
data = {
    'Usuario': ['user1', 'user2', 'user3', 'user4', 'user5'],
    'Accion': ['Combate', 'Exploración', 'Interaccion Social', 'Combate', 'Exploración'],
    'Duración': [120, 300, 180, 90, 240]
}

df = pd.DataFrame(data)

# Función de clasificación basada en reglas
def clasificar_accion(fila):
    # Regla para clasificar si la acción fue exitosa o fallida
    if fila['Accion'] == 'Interaccion Social' and fila['Duración'] >= 180:
        return 'Mensaje Enviado'
    
    elif fila['Accion'] == 'Exploración' and fila['Duración'] >= 300:
        return 'Descubrimiento'
    
    elif fila['Accion'] == 'Combate' and fila['Duración'] >= 120:
        return 'Victoria'
    
    else:
        return 'Derrota'

# Aplicar la función de clasificación
df['Resultado'] = df.apply(clasificar_accion, axis=1)

# Mostrar los resultados
print(df)