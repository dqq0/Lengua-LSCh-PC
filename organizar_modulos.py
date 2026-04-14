import os
import glob
import shutil
import unicodedata

# 1. Definimos las listas oficiales (limpias de tildes y en minúsculas)
MODULO_0 = [chr(i) for i in range(97, 123)] # Letras de 'a' hasta 'z'
MODULO_0.append("ll")
MODULO_0.append("rr")
MODULO_0.append("ch")
MODULO_0.append("n") # Incluímos la ñ por si acaso, que se mapea a n

MODULO_1 = [
    "hola", "chao", "adios", "buenos_dias", "buenas_tardes", "buenas_noches", 
    "como_estas", "bien", "mal", "mas_o_menos", "por_favor", "gracias", "de_nada", 
    "perdon", "disculpa", "nombre", "mi", "tu", "yo", "si", "no", "que", "como", 
    "donde", "cuando", "quien", "por_que", "entiendo", "no_entiendo", "saber", "no_saber", "ayuda"
]

MODULO_2 = [
    "mama", "papa", "hijo", "hija", "hermano", "hermana", "abuelo", "abuela", "tio", "tia", 
    "primo", "prima", "amigo", "amiga", "novio", "novia", "esposo", "esposa", "persona", "hombre", 
    "mujer", "nino", "nina", "joven", "adulto_mayor", "casa", "familia", "perro", "gato", "trabajo", 
    "estudiar", "colegio", "escuela", "universidad", "profesor", "profesora", "alumno", "alumna", 
    "medico", "doctor", "carabineros", "policia", "interprete", "sordo", "sorda", "suegro", "suegra", 
    "yerno", "nuero", "nuera", "cunado", "cunada", "matrimonio", "casado", "casada", "pareja", "pololo", "polola", "vecino"
]

MODULO_3 = [
    "comer", "beber", "tomar", "dormir", "jugar", "comprar", "querer", "necesitar", 
    "ir", "venir", "ver", "escuchar", "atender", "hablar", "caminar", "trabajar", "descansar"
]

MODULO_4 = [
    "hoy", "manana", "ayer", "ahora", "despues", "antes", "dia", "tarde", "noche", 
    "semana", "mes", "ano", "hora", "tiempo", "clima", "calendario"
]

MODULO_5 = [
    "iquique", "alto_hospicio", "playa", "mar", "oceano", "desierto", "sol", "calor", 
    "frio", "camanchaca", "puerto", "zona_franca", "zofri", "centro", "hospital", 
    "terminal", "buses", "almacen", "negocio", "supermercado", "farmacia", "banco", 
    "calle", "cerro", "feria", "cavancha", "comida", "agua", "pan", "bano", 
    "dinero", "plata", "precio", "cuanto_cuesta", "micro", "autobus", "auto", "taxi"
]

MODULO_6 = [
    "feliz", "triste", "enojado", "enfermo", "miedo", "amor", "gustar", "no_gustar", 
    "color", "numero", "diez", "cien", "mil", "cansado", "lsch"
]

# Mapa maestro
MAPA_MODULOS = {}
for palabra in MODULO_0: MAPA_MODULOS[palabra] = "modulo0"
for palabra in MODULO_1: MAPA_MODULOS[palabra] = "modulo1"
for palabra in MODULO_2: MAPA_MODULOS[palabra] = "modulo2"
for palabra in MODULO_3: MAPA_MODULOS[palabra] = "modulo3"
for palabra in MODULO_4: MAPA_MODULOS[palabra] = "modulo4"
for palabra in MODULO_5: MAPA_MODULOS[palabra] = "modulo5"
for palabra in MODULO_6: MAPA_MODULOS[palabra] = "modulo6"

def quitar_tildes(texto):
    texto = texto.lower()
    texto = texto.replace("ñ", "n")
    texto = ''.join((c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn'))
    return texto

def organizar_datasets():
    carpeta_dataset = "dataset"
    if not os.path.exists(carpeta_dataset):
        return

    # 1. Crear la carpeta del modulo0
    os.makedirs(os.path.join(carpeta_dataset, "modulo0"), exist_ok=True)

    archivos_npz = []
    for root, _, files in os.walk(carpeta_dataset):
        for file in files:
            if file.endswith('.npz'):
                archivos_npz.append(os.path.join(root, file))

    movidos = 0

    for path_npz in archivos_npz:
        nombre_base = os.path.basename(path_npz)
        partes = nombre_base.split('_')
        
        etiqueta = quitar_tildes(partes[0])
        etiqueta_alternativa = quitar_tildes(nombre_base).replace(".npz", "")

        destino_mod = None
        
        if etiqueta in MAPA_MODULOS:
            destino_mod = MAPA_MODULOS[etiqueta]
        else:
            destino_mod = "extras"
            
        if destino_mod:
            carpeta_destino = os.path.join(carpeta_dataset, destino_mod)
            os.makedirs(carpeta_destino, exist_ok=True)
            if os.path.abspath(os.path.dirname(path_npz)) != os.path.abspath(carpeta_destino):
                shutil.move(path_npz, os.path.join(carpeta_destino, nombre_base))
                json_path = path_npz.replace('.npz', '_metadata.json')
                if os.path.exists(json_path):
                    shutil.move(json_path, os.path.join(carpeta_destino, os.path.basename(json_path)))
                movidos += 1

    print(f"[OK] Se movieron {movidos} archivos del abecedario (Modulo 0) u otros rezagados a sus carpetas.")
            
if __name__ == "__main__":
    organizar_datasets()
