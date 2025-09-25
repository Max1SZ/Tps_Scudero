import math
#algoritmo foil

#Ejercicio 1
datos = [ 
{"edad": 22, "departamento": "IT", "nivel_educativo": "terciario", "en_formacion": True}, 
{"edad": 24, "departamento": "IT", "nivel_educativo": "universitario", "en_formacion": True}, 
{"edad": 21, "departamento": "RRHH", "nivel_educativo": "terciario", "en_formacion": True}, 
{"edad": 35, "departamento": "IT", "nivel_educativo": "universitario", "en_formacion": False}, 
{"edad": 40, "departamento": "Finanzas", "nivel_educativo": "maestría", "en_formacion": False}, 
{"edad": 29, "departamento": "RRHH", "nivel_educativo": "universitario", "en_formacion": False}, 
{"edad": 23, "departamento": "IT", "nivel_educativo": "terciario", "en_formacion": True}, 
{"edad": 38, "departamento": "Finanzas", "nivel_educativo": "universitario", "en_formacion": False}]

#identifica personas con edad entre 22 hasta 24 y nivel terciario 
for persona in datos:
    if persona["edad"] >= 21 and persona["edad"] <= 24 and persona["en_formacion"] == True:
        print(persona)

print("-"*50)
print("Ejercicio 1:\n*Regla inducida: si la edad está entre 21 y 24 " \
"y está en formación, entonces imprimir los datos." \
"\n*Nivel educativo comun: terciario" \
"\n*Edades que aparecen en postivio: 24,21,22,23")
print("-"*50) #descubri que \ sirve de delimitador para los comentarios.

#Ejercicio 2
#separar positivos y negativos
positivos = [d for d in datos if d["en_formacion"]]
negativos = [d for d in datos if not d["en_formacion"]]

#cuenta cuanto hay en cada uno
P, N = len(positivos), len(negativos)

#evaluar condición nivel_educativo == "terciario"
positivos_despues = [d for d in positivos if d["nivel_educativo"] == "terciario"]
negativos_despues = [d for d in negativos if d["nivel_educativo"] == "terciario"]

#aplica lo mismo que en el paso anterior
p, n = len(positivos_despues), len(negativos_despues)

#FOIL Gain
#replicamos la formula FOIL con math
frac1 = p / (p + n) if (p + n) > 0 else 0
frac2 = P / (P + N)
foil_gain = p * (math.log2(frac1) - math.log2(frac2))

#mostrar resultados
print("Ejercicio2:\nCondición: nivel_educativo == 'terciario'")
print(f"P (positivos antes) = {P}\n\
N (negativos antes) = {N}\n\
p (positivos después) = {p}\n\
n (negativos después) = {n}\n\
p / (p + n) = {frac1:.3f}\n\
P / (P + N) = {frac2:.3f}\n\
log2(p / (p + n)) = {math.log2(frac1):.3f}\n\
log2(P / (P + N)) = {math.log2(frac2):.3f}\n\
FOIL Gain = {foil_gain:.3f}")

