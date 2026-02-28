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
    
    npz_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
    
    print(f"\n📂 Cargando dataset desde: {dataset_dir}")
    print(f"   Archivos encontrados: {len(npz_files)}\n")
    
    for file in sorted(npz_files):
        filepath = os.path.join(dataset_dir, file)
        data = np.load(filepath)
        
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
    detector = SignLanguageDetector()
    
    print("\n" + "="*60)
    print("  SIGN LANGUAGE DETECTOR - LSCh")
    print("  Modo: Captura y Entrenamiento")
    print("="*60)
    print("\nControles:")
    print("  [R] - Iniciar grabación de seña")
    print("  [S] - Detener grabación y guardar")
    print("  [V] - Procesar video desde archivo")
    print("  [L] - Listar dataset actual")
    print("  [Q] - Salir")
    print("="*60 + "\n")
    
    # Intentar primero con la cámara índice 1 (generalmente la USB externa)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ℹ️ No se detectó cámara USB en el índice 1, intentando con la integrada (índice 0)...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Intento por defecto sin DSHOW
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
        
    print("✅ Cámara activada exitosamente. Abriendo ventana gráfica...")

    prev_time = 0
    empty_frames = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            empty_frames += 1
            if empty_frames % 50 == 0:
                print("⚠️ Advertencia: OpenCV no está logrando leer la imagen. ¿Quizás otra app (como Zoom o la app Cámara) la está usando?")
            continue
            
        empty_frames = 0

        # Procesar frame
        image, results = detector.process_frame(frame)
        image = detector.draw_landmarks(image, results)
        
        # Si está grabando, agregar frame al buffer
        if detector.is_recording:
            detector.add_frame_to_sequence(results)
            # Indicador visual de grabación
            cv2.circle(image, (50, 50), 15, (0, 0, 255), -1)
            cv2.putText(image, f"GRABANDO: {detector.sequence_label}", (70, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, f"Frames: {len(detector.sequence_buffer)}", (70, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        cv2.putText(image, f'FPS: {int(fps)}', (10, image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Sign Language Detector - LSCh', image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('r') and not detector.is_recording:
            label = input("\nNombre de la seña (ej: 'hola', 'gracias'): ").strip()
            if label:
                detector.start_recording(label)
                
        elif key == ord('s') and detector.is_recording:
            detector.stop_recording()
            
        elif key == ord('v'):
            # Procesar video desde archivo
            video_path = input("\nRuta del video (ej: videos/sena1.mp4): ").strip()
            label = input("Nombre de la seña: ").strip()
            if video_path and os.path.exists(video_path):
                detector.process_video_file(video_path, label=label if label else None)
            else:
                print("❌ Video no encontrado")
                
        elif key == ord('l'):
            # Listar dataset
            if os.path.exists("dataset"):
                files = [f for f in os.listdir("dataset") if f.endswith('.npz')]
                print(f"\n📂 Dataset ({len(files)} archivos):")
                for f in sorted(files)[:10]:  # Mostrar primeros 10
                    print(f"   - {f}")
                if len(files) > 10:
                    print(f"   ... y {len(files)-10} más")
            else:
                print("\n⚠️ No existe carpeta 'dataset'")
            
    cap.release()
    cv2.destroyAllWindows()
    detector.holistic.close()
    print("\n👋 ¡Hasta luego!")


if __name__ == "__main__":
    main()