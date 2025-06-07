# %% para que puedas correrlo con el jupyter y ver los diferentes audios
import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import librosa 
import librosa.display
import soundfile as sf
from IPython.display import display, Audio

audio, sr = sf.read('C:\\Users\\estudiante\\Desktop\\TareasPYTHON\\2025\\Voz\\marita3.wav')


print(audio)
print("Frecuencia de muestreo (Hz):", sr)
print("Cantidad total de muestras:", len(audio))
print("Duración (segundos):", len(audio) / sr)
print("Vector de la señal (primeros 10 valores):", audio[:10])
plt.plot(audio)
plt.show()
#reproduccion de array diferente frecuencia muestreo
#y = (audio*2**3).astype(np.int8)
#Audio(y,rate=sr)

# Audio original
print("Audio original:")
display(Audio(audio, rate=sr))

# Audio rápido
print("Audio más rápido:")
display(Audio(audio, rate=sr*2))

# Audio lento
print("Audio más lento:")
display(Audio(audio, rate=sr//2))

# Audio con menor profundidad de bits
print("Audio con menor calidad:")
y = (audio * 2**3).astype(np.int8)
display(Audio(y, rate=sr))
