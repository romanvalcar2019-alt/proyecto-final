from pathlib import Path
import cv2
import numpy as np
from PIL import Image
def resolver_ruta_imagen(nombre_buscado):
    "Busca una imagen en muestras/ ignorando mayúsculas y extensiones."
    base_dir = Path(__file__).resolve().parent.parent  # carpeta proyecto-final
    carpeta_muestras = base_dir / "muestras"

    if not carpeta_muestras.exists():
        raise FileNotFoundError(f"No existe la carpeta muestras en: {carpeta_muestras}")

    archivos = [
        p for p in carpeta_muestras.iterdir()
        if p.is_file() and not p.name.endswith("_deuteranopia.png")
    ]

    objetivo = nombre_buscado.strip().lower()

    # 1. Coincidencia exacta sin extensión (ej. 'imagen1')
    for archivo in archivos:
        if archivo.stem.lower() == objetivo:
            return str(archivo)

    # 2. Si empieza con el nombre buscado (ej. 'imagen1.png.png' o 'imagen 1')
    for archivo in archivos:
        if archivo.stem.lower().startswith(objetivo):
            return str(archivo)

    disponibles = [a.name for a in archivos]
    raise FileNotFoundError(
        f"No se encontró '{nombre_buscado}' en {carpeta_muestras}. "
        f"Archivos detectados: {disponibles}"
    )


def cargar_imagen(ruta_imagen):
    "Carga una imagen desde disco y la devuelve en RGB y escala de grises."
    imagen_bgr = cv2.imread(ruta_imagen)
    if imagen_bgr is None:
        raise FileNotFoundError(f"No se pudo abrir la imagen en: {ruta_imagen}")

    imagen_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    imagen_gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    return imagen_rgb, imagen_gris


def analizar_brillo_y_contraste(imagen_gris):
    "Calcula brillo promedio y contraste básico (escala 0 a 255)."
    brillo = float(np.mean(imagen_gris))
    contraste = float(np.std(imagen_gris))

    if brillo < 50:
        estado_brillo = "Muy oscura (puede dificultar ver detalles o UI)"
    elif brillo > 205:
        estado_brillo = "Muy luminosa (puede cansar la vista)"
    else:
        estado_brillo = "Brillo equilibrado"

    if contraste < 35:
        estado_contraste = "Bajo contraste (elementos poco diferenciados)"
    else:
        estado_contraste = "Contraste aceptable"

    return {
        "brillo": round(brillo, 2),
        "contraste": round(contraste, 2),
        "estado_brillo": estado_brillo,
        "estado_contraste": estado_contraste,
    }


def simular_daltonismo(imagen_rgb, tipo="deuteranopia"):
    "Aplica una transformación aproximada con NumPy para simular daltonismo."
    matrices = {
        "protanopia": np.array([
            [0.567, 0.433, 0.0],
            [0.558, 0.442, 0.0],
            [0.0, 0.242, 0.758],
        ]),
        "deuteranopia": np.array([
            [0.625, 0.375, 0.0],
            [0.7, 0.3, 0.0],
            [0.0, 0.3, 0.7],
        ]),
        "tritanopia": np.array([
            [0.95, 0.05, 0.0],
            [0.0, 0.433, 0.567],
            [0.0, 0.475, 0.525],
        ]),
    }

    matriz = matrices.get(tipo.lower(), matrices["deuteranopia"])
    img_float = imagen_rgb.astype(np.float32) / 255.0
    img_simulada = np.dot(img_float, matriz.T)
    img_simulada = np.clip(img_simulada * 255.0, 0, 255).astype(np.uint8)
    return img_simulada


def analizar_captura(nombre_base, tipo_daltonismo="deuteranopia", guardar_simulacion=True):
    "Ejecuta el análisis de accesibilidad sobre 'muestras/<nombre_base>'."
    ruta_imagen = resolver_ruta_imagen(nombre_base)
    imagen_rgb, imagen_gris = cargar_imagen(ruta_imagen)
    metricas = analizar_brillo_y_contraste(imagen_gris)

    print(f" Análisis de: {ruta_imagen} ")
    print(f"Brillo promedio: {metricas['brillo']} -> {metricas['estado_brillo']}")
    print(f"Contraste estimado: {metricas['contraste']} -> {metricas['estado_contraste']}")

    imagen_simulada = simular_daltonismo(imagen_rgb, tipo=tipo_daltonismo)
    print(f"Simulación aplicada: {tipo_daltonismo}")

    if guardar_simulacion:
        salida_bgr = cv2.cvtColor(imagen_simulada, cv2.COLOR_RGB2BGR)
        ruta_salida = f"muestras/{nombre_base}_{tipo_daltonismo}.png"
        cv2.imwrite(ruta_salida, salida_bgr)
        print(f"Imagen simulada guardada en: {ruta_salida}")

if __name__ == "__main__":
    tipos = ("protanopia", "deuteranopia", "tritanopia")
    for nombre in ("imagen1", "imagen2", "imagen3", "imagen4", "imagen5"):
        for tipo in tipos:
            try:
                analizar_captura(nombre, tipo_daltonismo=tipo, guardar_simulacion=True)
            except FileNotFoundError as error:
                print(error)