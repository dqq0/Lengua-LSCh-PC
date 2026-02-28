import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class PoseValidator:
    """
    Clase encargada de comparar vectores de MediaPipe y calcular el score de similitud.
    Aplica pesos específicos para la Lengua de Señas Chilena (LSCh).
    """
    def __init__(self):
        # En LSCh, las manos y la cara son críticas
        self.weights = {
            "pose": 0.15,
            "hands": 0.60,
            "face": 0.25
        }
        
    def calculate_frame_score(self, frame_student, frame_pattern):
        """
        Compara un solo frame (arreglo 1D de 1635 variables).
        Decodifica la estructura de la matriz:
        - Pose: [0:99]
        - Manos: [99:225]
        - Cara (Tesselation): [225:1629]
        - Cara (Features LSCh): [1629:1635]
        """
        if frame_student is None or frame_pattern is None:
            return 0.0, {"pose": 0, "hands": 0, "face": 0}
            
        # 1. Separar componentes (basado en extract_keypoints() de main.py)
        # Pose
        s_pose = frame_student[0:99]
        p_pose = frame_pattern[0:99]
        
        # Manos (Izquierda + Derecha)
        s_hands = frame_student[99:225]
        p_hands = frame_pattern[99:225]
        
        # Cara (Features críticas de LSCh al final del array)
        s_face_feat = frame_student[-6:]
        p_face_feat = frame_pattern[-6:]
        
        # 2. Calcular distancias (usando Euclidean distance + Normalización simple)
        # Una distancia de 0 es perfecto (score 100), mayor a 2 es muy malo (score 0)
        
        def _get_score(dist, threshold=2.0):
            return max(0, min(100, 100 * (1 - (dist / threshold))))
            
        dist_pose = euclidean(s_pose, p_pose) if np.any(s_pose) and np.any(p_pose) else 2.0
        score_pose = _get_score(dist_pose, 3.0) # Tolerancia mayor en cuerpo
        
        # Penar fuertemente si no hay manos detectadas cuando el patrón sí las tiene
        if np.any(p_hands) and not np.any(s_hands):
            score_hands = 0.0
        else:
            dist_hands = euclidean(s_hands, p_hands) if np.any(s_hands) else 0.0
            score_hands = _get_score(dist_hands, 1.5) # Muy estricto con las manos
            
        dist_face = euclidean(s_face_feat, p_face_feat) if np.any(s_face_feat) else 2.0
        score_face = _get_score(dist_face, 1.0) # Ultra estricto con las microexpresiones
        
        # 3. Score ponderado total
        total_score = (score_pose * self.weights["pose"] + 
                       score_hands * self.weights["hands"] + 
                       score_face * self.weights["face"])
                       
        breakdown = {
            "pose": score_pose,
            "hands": score_hands,
            "face": score_face
        }
        
        return total_score, breakdown

    def calculate_sequence_dtw(self, seq_student, seq_pattern):
        """
        Compara dos secuencias temporales de frames usando Dynamic Time Warping.
        Permite que el estudiante vaya más rápido o más lento que el video original.
        Devuelve el Average Score del alineamiento óptimo.
        """
        if len(seq_student) == 0 or len(seq_pattern) == 0:
            return 0.0, None
            
        # DTW retorna la distancia global y la ruta de alineación [(i, j)]
        # donde 'i' es el índice del estudiante y 'j' el del patrón
        distance, path = fastdtw(seq_student, seq_pattern, dist=euclidean)
        
        # Calcular el score frame a frame basado en el alineamiento óptimo de DTW
        frame_scores = []
        for i, j in path:
            score, _ = self.calculate_frame_score(seq_student[i], seq_pattern[j])
            frame_scores.append(score)
            
        # El score final es el promedio de los mejores alineamientos
        return np.mean(frame_scores), path
        
    def generate_feedback(self, breakdown, current_frame_student, current_frame_pattern):
        """
        Genera consejos en texto basados en qué parte de la matriz está fallando
        durante el frame actual.
        """
        feedback = []
        
        # Reglas estáticas del LSCh
        if breakdown["hands"] < 60:
            feedback.append("Corrige la posición de tus manos.")
            
        if breakdown["face"] < 50:
            # Analizar específicamente los features (los ultimos 6 valores del vector)
            s_face = current_frame_student[-6:]
            p_face = current_frame_pattern[-6:]
            
            # Recordar indices: 0:apertura boca, 1:ancho boca, 2:ceja izq, 3:ceja der
            mouth_s, mouth_p = s_face[0], p_face[0]
            eyebrows_s = (s_face[2] + s_face[3]) / 2
            eyebrows_p = (p_face[2] + p_face[3]) / 2
            
            if mouth_p > 0.3 and mouth_s < 0.1:
                feedback.append("Labios: abre más la boca (vocalización requerida)")
            if eyebrows_p > 0.15 and eyebrows_s < 0.05:
                feedback.append("Cejas: eleva más las cejas (es una pregunta)")
                
        if breakdown["pose"] < 40:
            feedback.append("Alinea mejor tu postura con la del modelo.")
            
        if len(feedback) == 0 and sum(breakdown.values())/3 > 80:
            feedback.append("¡Excelente!")
            
        return feedback
