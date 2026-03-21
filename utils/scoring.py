import numpy as np
from fastdtw import fastdtw

def euclidean(u, v):
    return np.linalg.norm(u - v)

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
        
    def _get_normalization_params(self, pose):
        """Calcula el centro y la escala a partir de los hombros (índices 11 y 12)."""
        if not np.any(pose):
            return np.zeros(3), 1.0
            
        p = pose.reshape(33, 3)
        ls = p[11] # Hombro izquierdo
        rs = p[12] # Hombro derecho
        
        origin = (ls + rs) / 2.0
        scale = np.linalg.norm(ls - rs)
        
        if scale < 0.05: # Evitar division por cero en frames fallidos
            scale = 1.0
            
        return origin, scale

    def _normalize_points(self, points, origin, scale):
        """Traslada al origen y escala independizando el tamaño de persona y distancia de la cámara."""
        if not np.any(points):
            return points
            
        p = points.reshape(-1, 3).copy()
        p = p - origin
        p = p / scale
        return p.flatten()
        
    def _normalize_sequence(self, sequence):
        """Aplica invarianza de traslación y escala a toda una secuencia resolviendo problemas de gente lejos o chica."""
        new_seq = []
        for frame in sequence:
            pose = frame[0:99]
            origin, scale = self._get_normalization_params(pose)
            
            n_pose = self._normalize_points(pose, origin, scale)
            n_hands = self._normalize_points(frame[99:225], origin, scale)
            n_face = frame[225:1629]
            feat = frame[1629:1635]
            
            new_frame = np.concatenate([n_pose, n_hands, n_face, feat])
            new_seq.append(new_frame)
        return new_seq

    def trim_idle_frames(self, sequence):
        """Recorta cuadros inactivos donde no hay manos detectadas (ignora esperas al inicio o final)."""
        active = []
        for i, frame in enumerate(sequence):
            hands = frame[99:225]
            if np.any(hands):
                active.append(i)
                
        if not active:
            return sequence
            
        start = max(0, active[0] - 2) # dejar un margen natural
        end = min(len(sequence), active[-1] + 3)
        return sequence[start:end]
        
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
        score_pose = _get_score(dist_pose, 8.0) # Compensar la escala de normalizacion espacial
        
        # Penar fuertemente si no hay manos detectadas cuando el patrón sí las tiene
        if np.any(p_hands) and not np.any(s_hands):
            score_hands = 0.0
        else:
            dist_hands = euclidean(s_hands, p_hands) if np.any(s_hands) else 0.0
            score_hands = _get_score(dist_hands, 4.0) # Tolerancia ajustada a normalización agnóstica de cámara
            
        dist_face = euclidean(s_face_feat, p_face_feat) if np.any(s_face_feat) else 2.0
        score_face = _get_score(dist_face, 1.5) # Estricto con las microexpresiones
        
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
        Compara secuencias temporalmente alineadas usando DTW e invarianza espacial.
        Permite tamaño/ubicación de usuario flexible, y recorta tiempos muertos naturales.
        """
        # 1. Recortar exceso de inactividad (bajar/subir brazos)
        s_trim = self.trim_idle_frames(seq_student)
        p_trim = self.trim_idle_frames(seq_pattern)
        
        if len(s_trim) == 0 or len(p_trim) == 0:
            return 0.0, None
            
        # 2. Normalizar espacialmente para independizar la distancia de la cámara y tamaño del usuario
        s_norm = self._normalize_sequence(s_trim)
        p_norm = self._normalize_sequence(p_trim)
        
        # 3. DTW sobre vectores ya normalizados
        distance, path = fastdtw(s_norm, p_norm, dist=euclidean)
        
        frame_scores = []
        for i, j in path:
            score, _ = self.calculate_frame_score(s_norm[i], p_norm[j])
            frame_scores.append(score)
            
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
