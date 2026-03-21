import cv2
import mediapipe as mp
import numpy as np

class AvatarReplicator:
    """
    Renderiza un Avatar esquelético estilizado con colores específicos 
    para repasar secuencias NPZ de la Lengua de Señas Chilena.
    """
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        
        # COLORES CUSTOMIZADOS BGR (Blue, Green, Red) dictados por el requerimiento
        # Cyan: (0, 200, 255) -> BGR: (255, 200, 0)
        self.left_hand_style = self.mp_drawing.DrawingSpec(color=(255, 200, 0), thickness=2, circle_radius=3)
        
        # Rojo coral: (255, 100, 100) -> BGR: (100, 100, 255)
        self.right_hand_style = self.mp_drawing.DrawingSpec(color=(100, 100, 255), thickness=2, circle_radius=3)
        
        # Azul: (100, 150, 255) -> BGR: (255, 150, 100)
        self.pose_style = self.mp_drawing.DrawingSpec(color=(255, 150, 100), thickness=3)
        
        # Cara contornos, dorado suave: (200, 180, 140) -> BGR: (140, 180, 200)
        self.face_contour_style = self.mp_drawing.DrawingSpec(color=(140, 180, 200), thickness=1)
        
        # Naranja brillante para cejas y cuello: (255, 200, 100) -> BGR: (100, 200, 255)
        self.accent_style = self.mp_drawing.DrawingSpec(color=(100, 200, 255), thickness=2)
        
        # Rosa para labios: (255, 150, 150) -> BGR: (150, 150, 255)
        self.lips_style = self.mp_drawing.DrawingSpec(color=(150, 150, 255), thickness=2)

    def reconstruct_landmarks_from_flat_array(self, frame_data):
        """
        Toma una fila del NPZ (1635 variables) y lo infla de vuelta a 
        objetos manipulables parecidos a NormalizedLandmarkList de MediaPipe.
        """
        # Estructura del array plano según main.py
        # [0:99] -> Pose (33 pts * 3)
        # [99:162] -> Left Hand (21 pts * 3)
        # [162:225] -> Right Hand (21 pts * 3)
        # [225:1629] -> Face (468 pts * 3)
        
        pose_data = frame_data[0:99].reshape(-1, 3)
        lh_data = frame_data[99:162].reshape(-1, 3)
        rh_data = frame_data[162:225].reshape(-1, 3)
        face_data = frame_data[225:1629].reshape(-1, 3)
        
        # --- CENTRADO AUTOMÁTICO HORIZONTAL ---
        if np.sum(pose_data) != 0:
            # Usamos el punto medio de los hombros (índices 11 y 12) para saber dónde está el usuario
            centro_hombros_x = (pose_data[11, 0] + pose_data[12, 0]) / 2.0
            # Queremos mover ese centro al medio de la pantalla (x: 0.5)
            offset_x = 0.5 - centro_hombros_x
            
            # Aplicamos este desplazamiento a TODOS los puntos extraídos
            pose_data[:, 0] += offset_x
            if np.sum(lh_data) != 0: lh_data[:, 0] += offset_x
            if np.sum(rh_data) != 0: rh_data[:, 0] += offset_x
            if np.sum(face_data) != 0: face_data[:, 0] += offset_x
        # --------------------------------------
        
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
                
            def HasField(self, field_name):
                return hasattr(self, field_name)
                
        class MockList:
            def __init__(self, arr):
                if np.sum(arr) == 0: # Array de ceros, no se detectó
                    self.landmark = []
                else:
                    self.landmark = [MockLandmark(row[0], row[1], row[2]) for row in arr]
                    
        return {
            'pose': MockList(pose_data) if np.sum(pose_data) != 0 else None,
            'left_hand': MockList(lh_data) if np.sum(lh_data) != 0 else None,
            'right_hand': MockList(rh_data) if np.sum(rh_data) != 0 else None,
            'face': MockList(face_data) if np.sum(face_data) != 0 else None
        }

    def render_frame(self, image, frame_data, is_npz=True, override_results=None):
        """
        Dibuja el marco estético con el estilo Custom del Replicador.
        """
        results = None
        if is_npz:
            # Recreamos el objeto simulado a partir de los números del NPZ almacenado
            # (El modelo ya fue ejecutado en el pasado)
            results = self.reconstruct_landmarks_from_flat_array(frame_data)
        else:
            # Usar los resultados 'en crudo' de un MP4
            results = override_results
            
            # Convirtiendo resultados reales al mismo formato que usa este módulo internamente
            pose_lm = results.pose_landmarks
            lh_lm = results.left_hand_landmarks
            rh_lm = results.right_hand_landmarks
            face_lm = results.face_landmarks
            results = {
                'pose': pose_lm,
                'left_hand': lh_lm,
                'right_hand': rh_lm,
                'face': face_lm
            }

        # 1. Dibujar Cara y Expresiones
        if results['face'] and results['face'].landmark:
            # Dibujar Malla Facial Completa (para la "máscara" y futuro humanoide)
            # Usando FACEMESH_TESSELATION en vez de CONTOURS
            self.mp_drawing.draw_landmarks(
                image, 
                results['face'], 
                self.mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=self.face_contour_style.color, thickness=1, circle_radius=0))
                
            # Extraer puntos clave de la cara para las cejas personalizadas
            h, w, _ = image.shape
            def get_face_pt(idx):
                lm = results['face'].landmark[idx]
                return (int(lm.x * w), int(lm.y * h))
                
            try:
                # --- Ceja Izquierda ---
                ceja_izq_int = get_face_pt(55) # Interior
                ceja_izq_ext = get_face_pt(105) # Exterior
                
                # --- Ceja Derecha ---
                ceja_der_int = get_face_pt(285) # Interior
                ceja_der_ext = get_face_pt(334) # Exterior
                
                # Dibujar las cejas como gruesas barras sólidas del MISMO color de la máscara
                color_cejas = self.face_contour_style.color
                cv2.line(image, ceja_izq_int, ceja_izq_ext, color_cejas, 8)
                cv2.line(image, ceja_der_int, ceja_der_ext, color_cejas, 8)
            except IndexError:
                pass

        # 2. Dibujar Pose Central (Estilo Stickman / Palito)
        if results['pose'] and results['pose'].landmark:
            
            # --- Dibujar Esqueleto con OpenCV Manualmente ---
            # Necesitamos calcular el punto medio para el cuello y la columna
            
            def get_pt(idx):
                lm = results['pose'].landmark[idx]
                # Convertir coordenadas normalizadas a píxeles
                h, w, _ = image.shape
                return (int(lm.x * w), int(lm.y * h))
                
            try:
                # Partes Superiores
                hombro_izq = get_pt(11)
                hombro_der = get_pt(12)
                nariz = get_pt(0)
                oreja_izq = get_pt(7)
                oreja_der = get_pt(8)
                
                # Caderas
                cadera_izq = get_pt(23)
                cadera_der = get_pt(24)
                
                # Brazos
                codo_izq = get_pt(13)
                muneca_izq = get_pt(15)
                codo_der = get_pt(14)
                muneca_der = get_pt(16)
                
                # Calcular centro de los hombros (base del cuello)
                centro_hombros = (int((hombro_izq[0] + hombro_der[0]) / 2),
                                  int((hombro_izq[1] + hombro_der[1]) / 2))
                                  
                # Calcular centro de las caderas (base de la columna)
                centro_caderas = (int((cadera_izq[0] + cadera_der[0]) / 2),
                                  int((cadera_izq[1] + cadera_der[1]) / 2))
                                  
                # --- LINEAS DEL STICKMAN ---
                color = self.pose_style.color
                thick = self.pose_style.thickness
                
                # Columna Central y Cuello Central
                cv2.line(image, centro_hombros, centro_caderas, color, thick)
                cv2.line(image, centro_hombros, nariz, color, thick)
                
                # Hombros y Clavículas
                cv2.line(image, hombro_izq, hombro_der, color, thick)
                
                # Brazo Izquierdo
                cv2.line(image, hombro_izq, codo_izq, color, thick)
                cv2.line(image, codo_izq, muneca_izq, color, thick)
                
                # Brazo Derecho
                cv2.line(image, hombro_der, codo_der, color, thick)
                cv2.line(image, codo_der, muneca_der, color, thick)
                
                # =========================================================
                # PUNTOS DE REFERENCIA EXPLÍCITOS PARA LSCh (Nodos de anclaje)
                # =========================================================
                color_nodo = (255, 255, 255) # Blanco para que destaquen
                color_orejas = self.face_contour_style.color
                
                # Orejas (Vital para señas como "escuchar", "audífono")
                cv2.circle(image, oreja_izq, 5, color_orejas, -1)
                cv2.circle(image, oreja_der, 5, color_orejas, -1)
                
                # Hombros/Clavículas (Vital para "yo", "responsabilidad")
                cv2.circle(image, hombro_izq, 4, color_nodo, -1)
                cv2.circle(image, hombro_der, 4, color_nodo, -1)
                
                # Caderas (Límite inferior del espacio de seña)
                cv2.circle(image, cadera_izq, 4, color_nodo, -1)
                cv2.circle(image, cadera_der, 4, color_nodo, -1)
                
                # Mentón / Papada (Calculado entre la nariz y base del cuello)
                # Vital para señas como "abuela", "viejo", "comer"
                menton = (int((nariz[0] + centro_hombros[0]) * 0.6), 
                          int((nariz[1] + centro_hombros[1]) * 0.6))
                cv2.circle(image, menton, 4, color_nodo, -1)
                
                # Estómago / Ombligo (Punto medio de la columna)
                # Vital para señas como "hambre", "embarazo", "dolor"
                estomago = (int((centro_hombros[0] + centro_caderas[0]) / 2),
                            int((centro_hombros[1] + centro_caderas[1]) / 2))
                cv2.circle(image, estomago, 6, (0, 255, 255), -1) # Amarillo claro
                
            except IndexError:
                # Fallback por si en algún cuadro MediaPipe no entrega los puntos
                pass

        # 3. Dibujar Manos con identidad de color propia
        if results['left_hand'] and results['left_hand'].landmark:
            self.mp_drawing.draw_landmarks(
                image,
                results['left_hand'],
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.left_hand_style,
                connection_drawing_spec=self.left_hand_style)
                
        if results['right_hand'] and results['right_hand'].landmark:
            self.mp_drawing.draw_landmarks(
                image,
                results['right_hand'],
                self.mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=self.right_hand_style,
                connection_drawing_spec=self.right_hand_style)
                
        return image
        
    def loop_playback(self, npz_path, label):
        """
        Bucle de reproduccion puramente visual para el archivo de entrada.
        """
        try:
            data = np.load(npz_path)
            sequence = data['sequence']
        except Exception as e:
            print(f"Error cargando NPZ: {e}")
            return
            
        total_frames = len(sequence)
        current_frame = 0.0
        is_paused = False
        slow_mode = False
        
        print("\n" + "="*50)
        print(f"🎬 Mostrando Patrón: {label}")
        print("  - [ESPACIO] Pausar / Continuar")
        print("  - [L] Activar/Desactivar Modo Lento (0.35x)")
        print("  - [E] Exportar Avatar a MP4 (Alta Calidad)")
        print("  - [A / D] Cuadro Anterior/Siguiente (Si está pausado)")
        print("  - [N / B] Siguiente/Anterior Seña en la categoría")
        print("  - [Q] Salir al menú principal")
        print("="*50 + "\n")

        while True:
            # Pantalla base oscura (fondo del avatar)
            frame_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # Dibujar avatar con interpolacion fluida Inteligente ("Motion Smoothing")
            idx_frame = int(current_frame)
            frac = current_frame - idx_frame
            idx_next = (idx_frame + 1) if (idx_frame + 1) < total_frames else idx_frame
            
            curr_seq = sequence[idx_frame]
            next_seq = sequence[idx_next]
            
            # Máscara para detectar puntos que no existen en MediaPipe
            mask_curr = curr_seq != 0.0
            mask_next = next_seq != 0.0
            mask_both = mask_curr & mask_next
            
            skeleton_data = np.zeros_like(curr_seq)
            
            # 1. Puntos presentes en ambas secuencias: Interpolar suavemente
            skeleton_data[mask_both] = (curr_seq[mask_both] * (1.0 - frac)) + (next_seq[mask_both] * frac)
            
            # 2. Puntos ausentes en 1 de los cuadros: Sin interpolar, se snap-ean
            mask_only_curr = mask_curr & ~mask_next
            mask_only_next = mask_next & ~mask_curr
            
            if frac < 0.5:
                skeleton_data[mask_only_curr] = curr_seq[mask_only_curr]
            else:
                skeleton_data[mask_only_next] = next_seq[mask_only_next]

            frame_img = self.render_frame(frame_img, skeleton_data, is_npz=True)
            
            # UI Overlay
            cv2.putText(frame_img, f"SEÑA: {label.upper()}", (30, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame_img, f"({idx_frame+1}/{total_frames})", (30, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        
            if slow_mode:
                cv2.putText(frame_img, "MODO LENTO FLUIDO (0.35x)", (950, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
            if is_paused:
                cv2.putText(frame_img, "PAUSADO - ESPACIO PARA REANUDAR", (400, 680), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow('Avatar Replicator Patrón', frame_img)
            
            key = cv2.waitKey(40 if not is_paused else 0) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return 'quit'
            elif key == ord('n'):
                return 'next'
            elif key == ord('b') or key == ord('p'):
                return 'prev'
            elif key == ord(' '):
                is_paused = not is_paused
            elif key == ord('l'):
                slow_mode = not slow_mode
            elif key == ord('e'):
                print(f"\n⏳ Exportando video HD de '{label}'... por favor espera.")
                self.export_to_mp4(sequence, label)
                cv2.imshow('Avatar Replicator Patrón', frame_img)
            elif is_paused and key == ord('d'):
                # Siguiente cuadro
                current_frame = (int(current_frame) + 1) % total_frames
            elif is_paused and key == ord('a'):
                # Cuadro anterior
                current_frame = (int(current_frame) - 1) % total_frames
                
            if not is_paused:
                current_frame += 0.35 if slow_mode else 1.0
                if current_frame >= total_frames:
                    current_frame = 0.0 # Loop back

    def export_to_mp4(self, sequence, label):
        """
        Exporta la secuencia del avatar a un archivo MP4 de alta calidad.
        """
        import os
        from datetime import datetime
        
        output_dir = "exportaciones_avatar"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_dir, f"{label}_avatar_HD_{timestamp}.mp4")
        
        # mp4v es ampliamente soportado en Windows y WhatsApp
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 10.0 # Mismo framerate target que se usa al grabar
        
        # 1280x720 es HD standard (720p)
        writer = cv2.VideoWriter(output_filename, fourcc, fps, (1280, 720))
        
        for skeleton_data in sequence:
            # Fondo negro
            frame_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Renderizamos limpio
            frame_img = self.render_frame(frame_img, skeleton_data, is_npz=True)
            
            # Etiqueta limpia
            cv2.putText(frame_img, f"{label.upper()}", (30, 60), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)
                        
            writer.write(frame_img)
            
        writer.release()
        print(f"✅ ¡Video HD guardado en la carpeta '{output_dir}'!")
