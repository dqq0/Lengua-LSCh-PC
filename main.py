import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from datetime import datetime

class SignLanguageDetector:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize Holistic model
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1, 
            smooth_landmarks=True,
            refine_face_landmarks=True)  # IMPORTANTE: Para iris tracking

        # Custom Drawing Specs
        self.hand_landmark_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
        self.hand_connection_style = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        self.pose_landmark_style = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=0, circle_radius=0)
        self.pose_connection_style = self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3)
        self.face_mesh_connection_style = self.mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=1)
        
        # Para almacenar secuencias de entrenamiento
        self.sequence_buffer = []  # Buffer temporal de frames
        self.sequence_label = None  # Etiqueta de la seña actual
        self.is_recording = False   # Estado de grabación

    # =========================================================================
    # MÉTODOS EXISTENTES (sin cambios)
    # =========================================================================
    
    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def draw_landmarks(self, image, results):
        # Face Mesh con transparencia
        if results.face_landmarks:
            overlay = image.copy()
            self.mp_drawing.draw_landmarks(
                overlay,
                results.face_landmarks,
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.face_mesh_connection_style)
            cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)
            
            # Dibujar cejas sólidas estilo Avatar para mantener la paridad visual
            h, w, _ = image.shape
            def get_face_pt(idx):
                lm = results.face_landmarks.landmark[idx]
                return (int(lm.x * w), int(lm.y * h))
                
            try:
                # Ceja Izquierda
                ceja_izq_int = get_face_pt(55)
                ceja_izq_ext = get_face_pt(105)
                # Ceja Derecha
                ceja_der_int = get_face_pt(285)
                ceja_der_ext = get_face_pt(334)
                
                # Color gris/blanco para que se camufle con la malla facial
                color_cejas = (220, 220, 220) # BGR
                cv2.line(image, ceja_izq_int, ceja_izq_ext, color_cejas, 6)
                cv2.line(image, ceja_der_int, ceja_der_ext, color_cejas, 6)
            except IndexError:
                pass

        # Pose (brazos y hombros)
        if results.pose_landmarks:
            upper_body_connections = frozenset([
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
            ])
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                upper_body_connections,
                landmark_drawing_spec=self.pose_landmark_style,
                connection_drawing_spec=self.pose_connection_style)
            
        # Manos
        self.mp_drawing.draw_landmarks(
            image,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.hand_landmark_style,
            connection_drawing_spec=self.hand_connection_style)
            
        self.mp_drawing.draw_landmarks(
            image,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.hand_landmark_style,
            connection_drawing_spec=self.hand_connection_style)
            
        return image

    # =========================================================================
    # NUEVOS MÉTODOS PARA EXTRACCIÓN DE CARACTERÍSTICAS
    # =========================================================================

    def extract_keypoints(self, results):
        """
        Extrae TODOS los keypoints numéricos de un frame.
        Retorna un array numpy con ~1650 valores.
        """
        # ========== 1. POSE (33 puntos × 3 coordenadas = 99 valores) ==========
        pose = np.zeros(33 * 3)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
        
        # ========== 2. MANOS (21 puntos × 3 × 2 manos = 126 valores) ==========
        left_hand = np.zeros(21 * 3)
        if results.left_hand_landmarks:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
            
        right_hand = np.zeros(21 * 3)
        if results.right_hand_landmarks:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        
        # ========== 3. CARA (468 puntos × 3 = 1404 valores) ==========
        face = np.zeros(468 * 3)
        if results.face_landmarks:
            face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]).flatten()
        
        # ========== 4. CARACTERÍSTICAS FACIALES ESPECÍFICAS LSCh ==========
        face_features = self.calculate_face_features(results.face_landmarks)
        
        # Concatenar todo: 99 + 126 + 1404 + 6 = 1635 valores
        return np.concatenate([pose, left_hand, right_hand, face, face_features])

    def calculate_face_features(self, face_landmarks):
        """
        Calcula características faciales GRAMATICALES para LSCh.
        Estas son CRÍTICAS para el reconocimiento correcto.
        
        Retorna array con 6 valores:
        [mouth_openness, mouth_width, eyebrow_raise_left, eyebrow_raise_right, 
         eye_openness_left, eye_openness_right]
        """
        if not face_landmarks:
            return np.zeros(6)
        
        lm = face_landmarks.landmark
        
        # ÍNDICES CLAVE DE MEDIAPIPE FACE MESH (468 puntos)
        # Labios
        UPPER_LIP = 13      # Labio superior centro
        LOWER_LIP = 14      # Labio inferior centro
        LEFT_LIP = 78       # Comisura izquierda
        RIGHT_LIP = 308     # Comisura derecha
        
        # Ojos (párpados)
        LEFT_EYE_TOP = 159
        LEFT_EYE_BOTTOM = 145
        RIGHT_EYE_TOP = 386
        RIGHT_EYE_BOTTOM = 374
        
        # Cejas
        LEFT_EYEBROW_TOP = 105
        LEFT_EYEBROW_BOTTOM = 70  # Cerca del ojo
        RIGHT_EYEBROW_TOP = 334
        RIGHT_EYEBROW_BOTTOM = 300
        
        # Referencia para normalización (distancia entre ojos es relativamente constante)
        LEFT_EYE_CENTER = 33
        RIGHT_EYE_CENTER = 263
        eye_distance = self._distance(lm[LEFT_EYE_CENTER], lm[RIGHT_EYE_CENTER])
        
        # Evitar división por cero
        if eye_distance < 0.001:
            eye_distance = 0.001
        
        # 1. APERTURA DE BOCA (crucial para expresiones en LSCh)
        # Normalizado por distancia entre ojos
        mouth_openness = self._distance(lm[UPPER_LIP], lm[LOWER_LIP]) / eye_distance
        
        # 2. ANCHO DE BOCA (sonrisa vs neutral)
        mouth_width = self._distance(lm[LEFT_LIP], lm[RIGHT_LIP]) / eye_distance
        
        # 3. ELEVACIÓN DE CEJAS IZQUIERDA (preguntas en LSCh)
        eyebrow_raise_left = (lm[LEFT_EYEBROW_BOTTOM].y - lm[LEFT_EYEBROW_TOP].y) / eye_distance
        
        # 4. ELEVACIÓN DE CEJAS DERECHA
        eyebrow_raise_right = (lm[RIGHT_EYEBROW_BOTTOM].y - lm[RIGHT_EYEBROW_TOP].y) / eye_distance
        
        # 5. APERTURA DE OJO IZQUIERDO
        eye_openness_left = self._distance(lm[LEFT_EYE_TOP], lm[LEFT_EYE_BOTTOM]) / eye_distance
        
        # 6. APERTURA DE OJO DERECHO
        eye_openness_right = self._distance(lm[RIGHT_EYE_TOP], lm[RIGHT_EYE_BOTTOM]) / eye_distance
        
        return np.array([
            mouth_openness,      # > 0.3 = boca abierta (sorpresa, vocalización)
            mouth_width,         # > 0.8 = sonrisa
            eyebrow_raise_left,  # > 0.15 = ceja elevada (pregunta)
            eyebrow_raise_right, # > 0.15 = ceja elevada
            eye_openness_left,   # < 0.05 = ojo cerrado (guiño)
            eye_openness_right   # < 0.05 = ojo cerrado
        ])

    def _distance(self, p1, p2):
        """Calcula distancia euclídea entre dos puntos 3D."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    # =========================================================================
    # NUEVOS MÉTODOS PARA PROCESAMIENTO DE VIDEOS (ENTRENAMIENTO)
    # =========================================================================

    def process_video_file(self, video_path, output_dir="dataset", label=None, fps_target=10):
        """
        Procesa un archivo de video completo y guarda los keypoints.
        
        Args:
            video_path: Ruta al video MP4
            output_dir: Carpeta donde guardar el dataset
            label: Nombre de la seña (ej: "hola", "gracias")
            fps_target: Frames por segundo a extraer (10 es buen balance)
        
        Returns:
            dict: Metadata del procesamiento
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video no encontrado: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Info del video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"Procesando video: {os.path.basename(video_path)}")
        print(f"Duración: {duration:.1f}s | Frames: {total_frames} | FPS: {video_fps}")
        print(f"Label: {label or 'SIN_ETIQUETA'}")
        print(f"{'='*50}\n")
        
        # Calcular cada cuántos frames tomar muestra
        frame_interval = max(1, int(video_fps / fps_target))
        
        sequence = []  # Secuencia de keypoints
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # Procesar solo cada N frames según fps_target
            if frame_count % frame_interval == 0:
                # Procesar con MediaPipe
                image, results = self.process_frame(frame)
                
                # Extraer keypoints
                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
                saved_count += 1
                
                # Visualizar progreso (opcional)
                display_image = self.draw_landmarks(image.copy(), results)
                progress = (frame_count / total_frames) * 100
                cv2.putText(display_image, f'Procesando: {progress:.1f}%', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Procesando video...', display_image)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Procesamiento cancelado.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return None
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Guardar la secuencia
        if len(sequence) > 0:
            metadata = self._save_sequence(sequence, output_dir, label, video_path)
            print(f"\n✅ Completado: {saved_count} frames guardados")
            print(f"   Secuencia shape: ({len(sequence)}, {len(sequence[0])})")
            return metadata
        
        return None

    def _save_sequence(self, sequence, output_dir, label, source_video):
        """
        Guarda una secuencia de keypoints en formato NPZ (optimizado).
        """
        # Crear estructura de carpetas
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar nombre único
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = os.path.splitext(os.path.basename(source_video))[0]
        label_str = label or "unknown"
        
        filename = f"{label_str}_{video_name}_{timestamp}.npz"
        filepath = os.path.join(output_dir, filename)
        
        # Guardar como NPZ (comprimido y eficiente)
        np.savez_compressed(
            filepath,
            sequence=np.array(sequence),  # Shape: (n_frames, n_features)
            label=label_str,
            source_video=source_video,
            timestamp=timestamp,
            n_frames=len(sequence),
            n_features=len(sequence[0])
        )
        
        # También guardar metadata en JSON para fácil lectura
        metadata = {
            "filename": filename,
            "filepath": filepath,
            "label": label_str,
            "source_video": source_video,
            "timestamp": timestamp,
            "n_frames": len(sequence),
            "n_features": len(sequence[0]),
            "feature_breakdown": {
                "pose": 99,
                "left_hand": 63,
                "right_hand": 63,
                "face": 1404,
                "face_features": 6
            }
        }
        
        meta_path = filepath.replace('.npz', '_metadata.json')
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

    def process_image_to_sequence(self, image_path, output_dir="dataset", label=None, num_frames=15):
        """
        Procesa una imagen estática (foto) y genera una secuencia artificial 
        repitiendo los keypoints para usarla en el validador como seña estática.
        """
        if not os.path.exists(image_path):
            print(f"❌ Imagen no encontrada: {image_path}")
            return None
            
        print(f"\nProcesando imagen estática: {os.path.basename(image_path)}")
        image = cv2.imread(image_path)
        if image is None:
            print("❌ No se pudo leer la imagen.")
            return None
            
        # Procesar con MediaPipe
        img_rgb, results = self.process_frame(image)
        
        # Extraer keypoints
        keypoints = self.extract_keypoints(results)
        
        if not results.left_hand_landmarks and not results.right_hand_landmarks and not results.pose_landmarks:
            print("⚠️ Advertencia: No se detectaron manos ni postura en la imagen.")
            
        # Duplicar los keypoints `num_frames` veces para engañar al sistema (secuencia estática)
        sequence = [keypoints for _ in range(num_frames)]
        
        # Guardar la secuencia
        metadata = self._save_sequence(sequence, output_dir, label, image_path)
        print(f"✅ Seña estática guardada: {metadata['filename']} ({num_frames} frames artificiales)")
        return metadata

    # =========================================================================
    # MÉTODOS PARA CAPTURA EN TIEMPO REAL (PRÁCTICA)
    # =========================================================================

    def start_recording(self, label):
        """Inicia grabación de secuencia desde webcam."""
        self.sequence_buffer = []
        self.sequence_label = label
        self.is_recording = True
        print(f"\n🔴 GRABANDO: '{label}' - Presiona 'S' para detener")

    def stop_recording(self, output_dir="dataset"):
        """Detiene grabación y guarda la secuencia."""
        self.is_recording = False
        if len(self.sequence_buffer) > 0:
            metadata = self._save_sequence(
                self.sequence_buffer, 
                output_dir, 
                self.sequence_label, 
                "webcam_capture"
            )
            print(f"✅ Grabación guardada: {metadata['filename']}")
            print(f"   Frames: {metadata['n_frames']}")
            return metadata
        print("⚠️ No se grabaron frames")
        return None

    def add_frame_to_sequence(self, results):
        """Agrega un frame al buffer de grabación."""
        if self.is_recording and results:
            keypoints = self.extract_keypoints(results)
            self.sequence_buffer.append(keypoints)


# =============================================================================
# FUNCIONES AUXILIARES PARA ENTRENAMIENTO
# =============================================================================

def load_dataset(dataset_dir="dataset"):
    """
    Carga todos los archivos NPZ del dataset y prepara para entrenamiento.
    Retorna X (secuencias) e y (labels).
    """
    sequences = []
    labels = []
    label_map = {}  # Mapeo nombre -> índice
    current_label_idx = 0
    
    npz_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith('.npz'):
                npz_files.append(os.path.join(root, file))
    
    print(f"\n📂 Cargando dataset desde: {dataset_dir}")
    print(f"   Archivos encontrados: {len(npz_files)}\n")
    
    for filepath in sorted(npz_files):
        data = np.load(filepath)
        file = os.path.basename(filepath)
        
        sequence = data['sequence']
        label = str(data['label'])
        
        # Asignar índice a label si es nuevo
        if label not in label_map:
            label_map[label] = current_label_idx
            current_label_idx += 1
        
        sequences.append(sequence)
        labels.append(label_map[label])
        
        print(f"   ✅ {file}: {label} ({len(sequence)} frames)")
    
    print(f"\n📊 Dataset cargado:")
    print(f"   Total secuencias: {len(sequences)}")
    print(f"   Clases: {len(label_map)} - {list(label_map.keys())}")
    
    return sequences, np.array(labels), label_map


def pad_sequences(sequences, max_length=None):
    """
    Normaliza secuencias a longitud fija (padding/truncado).
    Necesario para entrenar modelos LSTM/Transformer.
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            # Truncar
            padded.append(seq[:max_length])
        else:
            # Padding con ceros
            padding = np.zeros((max_length - len(seq), seq.shape[1]))
            padded.append(np.vstack([seq, padding]))
    
    return np.array(padded)


# =============================================================================
# MAIN - EJEMPLOS DE USO
# =============================================================================

def main():
    # Importaciones para los nuevos módulos integrados
    import sys
    
    # Manejar las importaciones asumiendo la estructura actual del proyecto
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from modules.avatar import AvatarReplicator
    from modules.validator import SignValidator

    detector = SignLanguageDetector()
    validator = SignValidator(detector)
    replicator = AvatarReplicator()
    
    def get_available_npz(dataset_dir="dataset"):
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        npz_files = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith('.npz'):
                    rel_path = os.path.relpath(os.path.join(root, file), dataset_dir)
                    npz_files.append(rel_path.replace('\\', '/'))
        return npz_files
    
    def get_label_from_arch(arch_rel_path):
        json_path = os.path.join("dataset", arch_rel_path.replace('.npz', '_metadata.json'))
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f).get('label', 'Desconocido')
            except:
                pass
        
        # Fallback: Quitar el timestamp, y tratar de extraer el nombre base completo
        base = os.path.basename(arch_rel_path)
        if '_202' in base:
            return base.split('_202')[0]
        return base.split('_')[0]
    
    while True:
        print("\n" + "="*60)
        print("  SISTEMA DE LENGUA DE SEÑAS CHILENA (LSCh)")
        print("="*60)
        print("  1. 🎥 Grabar nueva seña (Dataset)")
        print("  2. 🤖 Ver Avatar Patrón (Replicador)")
        print("  3. 🎓 Modo Práctica (Validador Tiempo Real)")
        print("  4. 📤 Procesar Videos a NPZ (Dataset)")
        print("  5. 🖼️ Procesar Foto a Seña Estática (Abecedario)")
        print("  Q. ❌ Salir")
        print("="*60)
        
        opcion = input("Elige una opción: ").strip().lower()
        
        if opcion == 'q':
            break
            
        elif opcion == '1':
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
                
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            label = input("\nNombre de la seña a grabar: ").strip()
            print("Presiona 'R' para grabar, 'S' para detener, 'Q' para volver al menú")
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success: continue
                
                image, results = detector.process_frame(frame)
                image = detector.draw_landmarks(image, results)
                
                if detector.is_recording:
                    detector.add_frame_to_sequence(results)
                    cv2.circle(image, (50, 50), 15, (0, 0, 255), -1)
                    cv2.putText(image, f"GRABANDO: {detector.sequence_label}", (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                cv2.imshow('Grabador LSCh', image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('r') and not detector.is_recording: detector.start_recording(label)
                elif key == ord('s') and detector.is_recording: detector.stop_recording()
                
            cap.release()
            cv2.destroyAllWindows()
            
        elif opcion == '2':
            archivos = get_available_npz()
            if not archivos:
                print("⚠️ No hay señas guardadas en el dataset.")
                continue

            # Agrupar por categorías (carpetas)
            categorias = {}
            for arch in archivos:
                # Extraer carpeta contenedora
                dir_name = os.path.dirname(arch)
                if not dir_name:
                    categoria = "General (Raíz)"
                else:
                    categoria = dir_name

                if categoria not in categorias:
                    categorias[categoria] = []
                categorias[categoria].append(arch)

            # --- 1. Seleccionar Categoría ---
            print("\nCategorías disponibles:")
            nombres_categorias = list(categorias.keys())
            for i, cat in enumerate(nombres_categorias):
                print(f"  {i+1}. {cat} ({len(categorias[cat])} señas)")

            try:
                idx_cat = int(input("\nElige el número de la categoría: ")) - 1
                if idx_cat < 0 or idx_cat >= len(nombres_categorias):
                    print("❌ Opción inválida.")
                    continue
                
                categoria_elegida = nombres_categorias[idx_cat]
                archivos_cat = categorias[categoria_elegida]

                # --- 2. Seleccionar Seña (dentro de la categoría) ---
                print(f"\nSeñas en '{categoria_elegida}':")
                for i, arch in enumerate(archivos_cat):
                    label = get_label_from_arch(arch)
                    print(f"  {i+1}. {label}")

                idx_sena = int(input("\nElige el número de la seña a ver: ")) - 1
                if 0 <= idx_sena < len(archivos_cat):
                    while True:
                        archivo_elegido = archivos_cat[idx_sena]
                        label_elegido = get_label_from_arch(archivo_elegido)
                        ruta = os.path.join("dataset", archivo_elegido)
                        
                        accion = replicator.loop_playback(ruta, label_elegido)
                        
                        if accion == 'next':
                            idx_sena = (idx_sena + 1) % len(archivos_cat)
                        elif accion == 'prev':
                            idx_sena = (idx_sena - 1) % len(archivos_cat)
                        else:
                            break
                else:
                    print("❌ Opción inválida.")
            except ValueError:
                print("❌ Entrada inválida. Por favor, ingresa un número.")
                
        elif opcion == '3':
            archivos = get_available_npz()
            if not archivos:
                print("⚠️ No hay señas guardadas para practicar.")
                continue
                
            # Agrupar por categorías (carpetas)
            categorias = {}
            for arch in archivos:
                dir_name = os.path.dirname(arch)
                if not dir_name:
                    categoria = "General (Raíz)"
                else:
                    categoria = dir_name

                if categoria not in categorias:
                    categorias[categoria] = []
                categorias[categoria].append(arch)

            # --- 1. Seleccionar Categoría ---
            print("\nCategorías disponibles para practicar:")
            nombres_categorias = list(categorias.keys())
            for i, cat in enumerate(nombres_categorias):
                print(f"  {i+1}. {cat} ({len(categorias[cat])} señas)")

            try:
                idx_cat = int(input("\nElige el número de la categoría: ")) - 1
                if idx_cat < 0 or idx_cat >= len(nombres_categorias):
                    print("❌ Opción inválida.")
                    continue
                
                categoria_elegida = nombres_categorias[idx_cat]
                archivos_cat = categorias[categoria_elegida]

                # --- 2. Seleccionar Seña (dentro de la categoría) ---
                print(f"\nSeñas en '{categoria_elegida}':")
                for i, arch in enumerate(archivos_cat):
                    label = get_label_from_arch(arch)
                    print(f"  {i+1}. {label}")
                    
                idx_sena = int(input("\nElige el número de la seña para practicar: ")) - 1
                if 0 <= idx_sena < len(archivos_cat):
                    archivo_elegido = archivos_cat[idx_sena]
                    label_elegido = get_label_from_arch(archivo_elegido)
                    ruta = os.path.join("dataset", archivo_elegido)
                    validator.start_practice_session(ruta, label_elegido)
                else:
                     print("❌ Opción inválida.")
            except ValueError:
                 print("❌ Entrada inválida. Por favor, ingresa un número.")
                 
        elif opcion == '4':
            input_path = input("\nRuta del video (MP4, MOV, etc) o carpeta (ej: videos/tiempo): ").strip()
            
            if not input_path or not os.path.exists(input_path):
                print("❌ Ruta no encontrada.")
                continue
                
            if os.path.isdir(input_path):
                print(f"\n📂 Buscando videos en la carpeta: {input_path}")
                video_files = []
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                            video_files.append(os.path.join(root, file))
                            
                if not video_files:
                    print(f"⚠️ No se encontraron videos en {input_path}.")
                    continue
                    
                print(f"✅ Se encontraron {len(video_files)} videos. Iniciando procesamiento en lote...")
                for vp in video_files:
                    base_name = os.path.splitext(os.path.basename(vp))[0]
                    rel_dir = os.path.relpath(os.path.dirname(vp), input_path)
                    
                    base_folder = os.path.basename(os.path.normpath(input_path))
                    if rel_dir == '.':
                        out_dir = os.path.join("dataset", base_folder)
                    else:
                        out_dir = os.path.join("dataset", base_folder, rel_dir)
                        
                    detector.process_video_file(vp, output_dir=out_dir, label=base_name)
                    
            elif os.path.isfile(input_path) and input_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                label = input("Nombre de la seña (deja en blanco para usar nombre del archivo): ").strip()
                if not label:
                    label = os.path.splitext(os.path.basename(input_path))[0]
                detector.process_video_file(input_path, output_dir="dataset", label=label)
            else:
                print("❌ El archivo provisto no es un video soportado (.mp4, .mov, .avi, .mkv).")

        elif opcion == '5':
            input_path = input("\nRuta de la foto (.jpg, .png) o carpeta (ej: abecedario/): ").strip()
            
            if not input_path or not os.path.exists(input_path):
                print("❌ Ruta no encontrada.")
                continue
                
            if os.path.isdir(input_path):
                print(f"\n📂 Buscando imágenes en la carpeta: {input_path}")
                img_files = []
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_files.append(os.path.join(root, file))
                            
                if not img_files:
                    print(f"⚠️ No se encontraron imágenes en {input_path}.")
                    continue
                    
                print(f"✅ Se encontraron {len(img_files)} imágenes. Iniciando procesamiento en lote...")
                for ip in img_files:
                    label = os.path.splitext(os.path.basename(ip))[0]
                    base_folder = os.path.basename(os.path.normpath(input_path))
                    rel_dir = os.path.relpath(os.path.dirname(ip), input_path)
                    
                    if rel_dir == '.':
                        out_dir = os.path.join("dataset", base_folder)
                    else:
                        out_dir = os.path.join("dataset", base_folder, rel_dir)
                        
                    detector.process_image_to_sequence(ip, output_dir=out_dir, label=label, num_frames=15)
                    
            elif os.path.isfile(input_path) and input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                label = input("Nombre de la letra/seña (deja en blanco para usar nombre del archivo): ").strip()
                if not label:
                    label = os.path.splitext(os.path.basename(input_path))[0]
                
                # Por defecto lo guardamos en dataset/abecedario si es solo una foto
                out_dir = os.path.join("dataset", "abecedario")
                detector.process_image_to_sequence(input_path, output_dir=out_dir, label=label, num_frames=15)
            else:
                 print("❌ El archivo provisto no es una imagen soportada (.jpg, .png).")

    detector.holistic.close()
    print("\n👋 ¡Hasta luego!")

if __name__ == "__main__":
    main()