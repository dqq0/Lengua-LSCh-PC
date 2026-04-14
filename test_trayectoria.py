import cv2
import mediapipe as mp
import numpy as np
import os

class OclusionSolver:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        # Usamos model_complexity=2 que es el más pesado y preciso de MediaPipe
        # Subimos el tracking_confidence para forzar al modelo a "no soltar" la mano
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.8, 
            model_complexity=2,
            smooth_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.hand_connections = self.mp_holistic.HAND_CONNECTIONS
        
        # Historial para calcular la Inercia/Trayectoria
        self.history = {
            'Left': {'points': None, 'velocity': None, 'missing_frames': 0},
            'Right': {'points': None, 'velocity': None, 'missing_frames': 0}
        }
        
        # Si desaparece, intentará predecir hasta por "X" frames 
        # antes de rendirse y aceptar que la persona bajó los brazos.
        self.MAX_MISSING_FRAMES = 15 

    def _get_points_array(self, hand_landmarks):
        if not hand_landmarks: 
            return None
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

    def predecir_mano(self, lado, landmarks_actuales, pose_landmarks):
        """Ancla la mano perdida a la muñeca del brazo de MediaPipe, o usa trayectoria física"""
        pts = self._get_points_array(landmarks_actuales)
        hist = self.history[lado]
        
        if pts is not None:
            # ¡Mano Encontrada! Actualizar velocidad
            if hist['points'] is not None:
                hist['velocity'] = pts - hist['points']
            hist['points'] = pts
            hist['missing_frames'] = 0
            return pts, True  
        else:
            # ¡Mano Perdida! 
            if hist['points'] is not None and hist['missing_frames'] < self.MAX_MISSING_FRAMES:
                
                # ESTRATEGIA 1: Anclaje al Brazo (Kinematic Anchoring)
                # No tiene sentido perder la mano si seguimos viendo el brazo
                if pose_landmarks:
                    # El punto 15 es la muñeca izq, el 16 la muñeca der en el Pose tracking general
                    pose_idx = 15 if lado == 'Left' else 16
                    wrist_pose = pose_landmarks.landmark[pose_idx]
                    
                    # Si el brazo tiene buena visibilidad
                    if wrist_pose.visibility > 0.4:
                        wrist_pose_pt = np.array([wrist_pose.x, wrist_pose.y, wrist_pose.z])
                        last_wrist_hand = hist['points'][0] # El punto 0 es la raíz de la mano
                        
                        # Calculamos hacia dónde se movió el brazo entero
                        movimiento_brazo = wrist_pose_pt - last_wrist_hand
                        
                        # Movemos toda la mano (los 21 dedos) atada a ese movimiento
                        predicted_pts = hist['points'] + movimiento_brazo
                        
                        hist['points'] = predicted_pts
                        hist['velocity'] = movimiento_brazo # Guardamos la inercia
                        hist['missing_frames'] += 1
                        return predicted_pts, False

                # ESTRATEGIA 2: Fricción e inercia (Si hasta el brazo se escondió detrás del cuerpo)
                if hist['velocity'] is not None:
                    predicted_pts = hist['points'] + hist['velocity']
                    hist['velocity'] *= 0.85 
                    hist['points'] = predicted_pts
                    hist['missing_frames'] += 1
                    return predicted_pts, False
                    
            return None, False

    def dibujar_mano_predicha(self, image, points, es_real):
        """Dibuja la mano Verde si es real, Naranja si es predicción por trayectoria"""
        if points is None: return
        
        h, w, c = image.shape
        
        # Colores (BGR en OpenCV)
        color_punto = (0, 255, 0) if es_real else (0, 165, 255) # Verde / Naranja
        color_linea = (0, 200, 0) if es_real else (0, 100, 255)
        
        puntos_2d = []
        for p in points:
            x, y = int(p[0] * w), int(p[1] * h)
            puntos_2d.append((x, y))
            cv2.circle(image, (x, y), 3, color_punto, -1)
            
        for connection in self.hand_connections:
            idx1, idx2 = connection
            try:
                p1, p2 = puntos_2d[idx1], puntos_2d[idx2]
                cv2.line(image, p1, p2, color_linea, 2)
            except IndexError:
                pass


    def probar_video(self, video_path, slow_down_factor=3):
        if not os.path.exists(video_path):
            print(f"❌ No se encontró el video: {video_path}")
            return
            
        cap = cv2.VideoCapture(video_path)
        fps_original = cap.get(cv2.CAP_PROP_FPS)
        
        print("\n" + "="*50)
        print("  MODO TEST: PREDICCIÓN DE TRAYECTORIA EN OCLUSIÓN")
        print("="*50)
        print(" - Manos VERDES: Detectadas correctamente")
        print(" - Manos NARANJAS: Mano escondida (Predicción IA guiada por inercia)")
        print(" - Control: Usa la tecla [ESPACIO] para pausar/reanudar y ver el detalle.")
        print(" - Salir: Tecla [Q]")
        print("="*50 + "\n")
        
        pausado = False

        while cap.isOpened():
            if not pausado:
                success, frame = cap.read()
                if not success:
                    # Bucle eterno para analizar si se corta al final
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.history = {k: {'points': None, 'velocity': None, 'missing_frames': 0} for k in self.history}
                    continue
                
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Predecir e Interpolar atando la mano al brazo
                pts_izq, izq_real = self.predecir_mano('Left', results.left_hand_landmarks, results.pose_landmarks)
                pts_der, der_real = self.predecir_mano('Right', results.right_hand_landmarks, results.pose_landmarks)
                
                # Dibujar
                self.dibujar_mano_predicha(image, pts_izq, izq_real)
                self.dibujar_mano_predicha(image, pts_der, der_real)
                
                # Interfaz Gráfica
                if not izq_real and pts_izq is not None:
                    cv2.putText(image, "INTUYENDO MANO IZQ OCLUIDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                if not der_real and pts_der is not None:
                    cv2.putText(image, "INTUYENDO MANO DER OCLUIDA", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow('Analizador de Oclusion', image)
            
            # Cámara lenta o pausa
            delay = int((1000 / fps_original) * slow_down_factor) if not pausado else 0
            
            key = cv2.waitKey(delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                pausado = not pausado
                
        cap.release()
        cv2.destroyAllWindows()
        self.holistic.close()

if __name__ == "__main__":
    solver = OclusionSolver()
    ruta_input = input("Ingresa la ruta del video con problemas de cruce (ej. videos/verbos/comer.mp4): ")
    # Limpiamos espacios o comillas accidentales que a veces pone la terminal al copiar y pegar
    ruta = ruta_input.strip().strip('"').strip("'")
    solver.probar_video(ruta)
