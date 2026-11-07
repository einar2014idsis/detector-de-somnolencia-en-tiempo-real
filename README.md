# Detector de somnolencia en tiempo real

Aplicación de escritorio en Python que procesa imágenes de una cámara para observar señales faciales asociadas a somnolencia. Combina OpenCV, MediaPipe Face Mesh y una interfaz Tkinter para visualizar métricas, emitir alertas y registrar eventos.

## Funcionalidades

- Captura de cámara en tiempo real y detección de un rostro con MediaPipe Face Mesh.
- Cálculo de relaciones de apertura de ojos (EAR) y boca (MAR) a partir de puntos faciales.
- Calibración inicial del umbral de ojos y opción de recalibrar durante la sesión.
- Detección de cierre prolongado de ojos y bostezos, con alertas sonoras cuando el audio está disponible.
- Contadores de parpadeos, bostezos y alertas; estimación heurística de fatiga y visualización de FPS.
- Ajustes de duración de ojos cerrados, sensibilidad de bostezo, volumen y malla facial.
- Registro local de eventos en `registro_eventos.csv`, con fecha, hora, EAR, MAR e indicador de fatiga.

El indicador de fatiga es una heurística del proyecto, no una medición clínica ni una garantía de seguridad al conducir. La iluminación, la cámara y la detección del rostro afectan el resultado.

## Tecnologías y estructura

| Componente | Uso |
|---|---|
| Python y Tkinter | Interfaz y controles de sesión |
| OpenCV | Captura y visualización de video |
| MediaPipe Face Mesh | Puntos faciales para las métricas |
| NumPy | Cálculos geométricos |
| Pygame | Reproducción de `alerta.wav` |

`proyecto_somnolencia.py` contiene la interfaz, captura, cálculo y registro; `requirements.txt` fija versiones de dependencias; `alerta.wav` es el sonido de alerta. La captura utiliza `cv2.CAP_DSHOW`, por lo que el flujo está orientado a Windows.

## Instalación y ejecución

Necesitas Windows, Python compatible con las versiones de `requirements.txt`, Tkinter, una cámara y un dispositivo de audio. Desde la carpeta del proyecto:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe proyecto_somnolencia.py
```

Selecciona **iniciar** y mantén los ojos abiertos durante la calibración inicial. La ventana de video puede cerrarse con `q`; la interfaz también permite detener o finalizar la sesión. Si el audio no está disponible, la interfaz lo indica y la detección puede continuar sin sonido.

El archivo CSV se crea junto al programa cuando se registra un evento; ten en cuenta la privacidad de estos datos al compartirlo.
