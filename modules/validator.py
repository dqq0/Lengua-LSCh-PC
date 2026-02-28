import cv2
import time
import numpy as np
from modules.avatar import AvatarReplicator
from utils.scoring import PoseValidator
import mediapipe as mp

class SignValidator:
    def __init__(self, main_detector):
        self.detector = main_detector # Instancia de SignLanguageDetector del main.py
        self.avatar = AvatarReplicator()
        self.validator = PoseValidator()
        
    def start_practice_session(self, npz_path, label):
        """
        Inicia un loop de práctica donde el estudiante debe imitar al Avatar.
        """
        try:
            data = np.load(npz_path)
            pattern_sequence = data['sequence']
        except Exception as e:
            print(f"Error cargando NPZ para práctica: {e}")
            return
            
        print("\n" + "="*50)
        print(f"🎓 SESIÓN DE PRÁCTICA: {label.upper()}")
        print("  - [ESPACIO] Iniciar captura de tu intento")
        print("  - [Q] Salir al menú")
        print("="*50 + "\n")
            
        # Intentar conectar cámara USB o integrada
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        total_frames = len(pattern_sequence)
        current_frame = 0
        state = "WAITING" # WAITING, RECORDING, RESULTS
        student_sequence = []
        feedback_msgs = []
        final_score = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
                
            frame = cv2.flip(frame, 1) # Efecto espejo natural para practicar
            
            # 1. Crear lienzo base para Split-Screen
            # 720 alto x 1280 ancho
            split_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
            
            # --- MITAD IZQUIERDA: AVATAR PATRÓN ---
            avatar_bg = np.zeros((720, 640, 3), dtype=np.uint8)
            if state in ["WAITING", "RECORDING"]:
                # Renderizar frame del patrón en la mitad izquierda
                idx_patron = int(current_frame)
                avatar_bg = self.avatar.render_frame(avatar_bg, pattern_sequence[idx_patron], is_npz=True)
            
            # Texto mitad izquierda
            cv2.putText(avatar_bg, "PATRON A SEGUIR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(avatar_bg, f"{label.upper()} ({int(current_frame)+1}/{total_frames})", (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 200, 0), 2)
            
            # Colocar mitad izquierda
            split_screen[:, :640] = avatar_bg
            
            # --- MITAD DERECHA: ALUMNO EN VIVO ---
            # Evitar `cv2.resize` para no aplastar la imagen y romper el Face Tracking
            # En su lugar, recortar el centro de la imagen (640 pixeles de ancho)
            h, w = frame.shape[:2]
            centro_x = w // 2
            student_bg = frame[:, centro_x - 320 : centro_x + 320].copy()
            
            # Procesar frame del estudiante con MediaPipe
            image, results = self.detector.process_frame(student_bg)
            
            if state == "WAITING":
                # Solo dibujar landmarks visuales, no grabar
                student_bg = self.detector.draw_landmarks(student_bg.copy(), results)
                cv2.putText(student_bg, "TU CAMARA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                
                # Instrucción parpadeante
                if int(time.time() * 2) % 2 == 0:
                    cv2.putText(student_bg, "PRESIONA ESPACIO PARA INTENTAR", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
            elif state == "RECORDING":
                # Grabar estudiante y comparar en tiempo real (feedback instantáneo)
                student_bg = self.detector.draw_landmarks(student_bg.copy(), results)
                keypoints = self.detector.extract_keypoints(results)
                student_sequence.append(keypoints)
                
                # Feedback del frame actual
                idx_patron = int(current_frame)
                score, breakdown = self.validator.calculate_frame_score(keypoints, pattern_sequence[idx_patron])
                feedback_msgs = self.validator.generate_feedback(breakdown, keypoints, pattern_sequence[idx_patron])
                
                # Mostrar feedback instantáneo
                color = (0, 255, 0) if score > 70 else (0, 100, 255) if score > 40 else (0, 0, 255)
                cv2.putText(student_bg, f"Precision actual: {int(score)}%", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
                
                y_offset = 80
                for msg in feedback_msgs:
                    cv2.putText(student_bg, f"- {msg}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)
                    y_offset += 30
                    
                # Avanzar frame a velocidad 0.75x para facilitar la práctica
                current_frame += 0.75
                if int(current_frame) >= total_frames:
                    state = "RESULTS"
                    
            elif state == "RESULTS":
                # Calcular DTW final
                if len(student_sequence) > 0 and final_score == 0:
                    print("Calculando DTW para el intento...")
                    final_score, path = self.validator.calculate_sequence_dtw(student_sequence, pattern_sequence)
                    
                student_bg.fill(30) # Gris oscuro
                cv2.putText(student_bg, "PUNTAJE FINAL", (200, 150), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2)
                
                color_score = (0, 255, 0) if final_score > 75 else (0, 160, 255) if final_score > 50 else (0, 0, 255)
                cv2.putText(student_bg, f"{int(final_score)}/100", (220, 230), cv2.FONT_HERSHEY_DUPLEX, 2, color_score, 3)
                
                if final_score > 80:
                    msg = "¡LO LOGRASTE!"
                elif final_score > 50:
                    msg = "¡CASI! INTENTA DE NUEVO."
                else:
                    msg = "REVISA EL PATRON E INTENTA DE NUEVO."
                    
                cv2.putText(student_bg, msg, (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(student_bg, "[ESPACIO] Reintentar  |  [Q] Menú", (150, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
                
            # Colocar mitad derecha
            split_screen[:, 640:] = student_bg
            
            # Línea divisoria central
            cv2.line(split_screen, (640, 0), (640, 720), (255, 255, 255), 4)
            
            cv2.imshow('Sign Language Validator', split_screen)
            
            key = cv2.waitKey(40) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if state == "WAITING":
                    state = "RECORDING"
                    current_frame = 0
                    student_sequence = []
                elif state == "RESULTS":
                    state = "WAITING"
                    final_score = 0
                    current_frame = 0
                    
        cap.release()
        cv2.destroyAllWindows()
