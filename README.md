# Detector y Avatar de Lengua de Señas Chilena (LSCh)

Este proyecto te permite grabar, ver y practicar señas a través de un Avatar esquelético estilizado.

## Requisitos
- Tener una cámara conectada.
- Python 3.8 o superior.

## Instalación
Para instalar lo necesario, abre tu terminal y ejecuta:

```bash
pip install -r requirements.txt
```

## ¿Cómo usarlo?
Solo tienes que ejecutar el archivo principal:

```bash
python main.py
```

Al abrirlo, verás un menú con 4 opciones principales. Solo escribe el **número** y presiona 'Enter':

1. **Grabar nueva seña (Dataset):** Se encenderá tu cámara. Puedes grabarte haciendo una seña y el sistema guardará tu movimiento.
2. **Ver Avatar Patrón (Replicador):** Muestra todas las señas que ya están procesadas y guardadas (el dataset). Podrás ver al Avatar haciendo la seña.
3. **Modo Práctica (Validador):** Tu cámara se encenderá a un lado de la pantalla y el Avatar al otro. Tendrás que imitar el movimiento exacto del Avatar para practicar la seña.
4. **Procesar MP4 a NPZ (Dataset):** Puedes transformar un video `.mp4` normal de tu computador directamente al formato esqueleto del Avatar de forma masiva (solo le entregas la ruta de la carpeta).

---

### 🎮 Controles de Teclado (Al reproducir el Avatar)

Cuando entras a la **Opción 2** para repasar alguna seña que ya está guardada, puedes utilizar tu teclado para controlar al Avatar:

*   **ESPACIO**: Sirve para **Pausar** y **Continuar** la reproducción del Avatar.
*   **E**: Te permite **Exportar** lo que el Avatar está haciendo y te generará automáticamente un video HD (`.mp4`) en tu computador de muy alta calidad y con el fondo limpio en negro.
*   **N**: Te permite saltar a la **Siguiente** seña rápidamente sin salir de la ventana.
*   **B** o **P**: Te permite retroceder a la seña **Anterior** rápidamente sin salir de la ventana.

*Si tienes el Avatar pausado (con ESPACIO), puedes usar:*
*   **D**: Avanza el Avatar exactamente un cuadro.
*   **A**: Retrocede el Avatar exactamente un cuadro.  

*(Ideal para estudiar la estructura y gramática de cómo se hace la seña paso por paso).*

---
*Para salir de cualquier reproductor en cualquier momento, presiona la tecla **Q**.*
