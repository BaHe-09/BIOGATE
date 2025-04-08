import cv2
import numpy as np
import psycopg2
from psycopg2.extras import Json
from keras_facenet import FaceNet
from ultralytics import YOLO
from dotenv import load_dotenv
from datetime import datetime
import time

load_dotenv()

class ReconocimientoFacial:
    def __init__(self):
        self.camara = cv2.VideoCapture(0)
        self.yolo = YOLO('models/yolov8n-face.pt')
        self.facenet = FaceNet()
        self.db_conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
        self.umbral = 0.7
        self.dispositivo_id = self.registrar_dispositivo()

    def registrar_dispositivo(self):
        """Registra el dispositivo de cámara local"""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO dispositivos 
                    (nombre, tipo, ubicacion, estado)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (nombre) DO UPDATE SET
                        tipo = EXCLUDED.tipo,
                        estado = EXCLUDED.estado
                    RETURNING id_dispositivo
                """, (
                    'Cámara Local Docker',
                    'Cámara',
                    'Contenedor Docker',
                    'Activo'
                ))
                self.db_conn.commit()
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error al registrar dispositivo: {str(e)}")
            return None

    def generar_embedding(self, rostro):
        """Genera embedding facial"""
        rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
        rostro_redimensionado = cv2.resize(rostro_rgb, (160, 160))
        return self.facenet.embeddings(np.expand_dims(rostro_redimensionado, axis=0))[0]

    def buscar_en_db(self, embedding):
        """Busca coincidencias en la base de datos"""
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT p.id_persona, p.nombre, p.apellido_paterno, 
                       p.apellido_materno, 1 - (v.vector <=> %s::vector) as similitud
                FROM vectores_identificacion v
                JOIN personas p ON v.id_persona = p.id_persona
                WHERE 1 - (v.vector <=> %s::vector) > %s
                ORDER BY similitud DESC
                LIMIT 1
            """, (embedding.tolist(), embedding.tolist(), self.umbral))
            return cursor.fetchone()

    def registrar_acceso(self, persona_id, confianza, frame):
        """Registra el acceso en la base de datos"""
        try:
            # Guardar imagen temporalmente
            temp_file = f"/tmp/acceso_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(temp_file, frame)
            
            with open(temp_file, 'rb') as f:
                imagen_bytes = f.read()
            
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO historial_accesos 
                    (id_persona, id_dispositivo, resultado, confianza, foto_url, metadatos)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (
                    persona_id,
                    self.dispositivo_id,
                    'Éxito' if persona_id else 'Fallo',
                    float(confianza),
                    f"data:image/jpeg;base64,{imagen_bytes}",
                    Json({
                        "sistema": "Docker",
                        "modelo": "YOLOv8-Face + FaceNet",
                        "timestamp": datetime.now().isoformat()
                    })
                ))
                self.db_conn.commit()
        except Exception as e:
            print(f"Error al registrar acceso: {str(e)}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def procesar_frame(self, frame):
        """Procesa un frame de la cámara"""
        resultados = self.yolo(frame)
        
        if resultados and len(resultados[0].boxes) > 0:
            for box in resultados[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                rostro = frame[y1:y2, x1:x2]
                
                if rostro.size == 0:
                    continue
                
                try:
                    embedding = self.generar_embedding(rostro)
                    coincidencia = self.buscar_en_db(embedding)
                    
                    if coincidencia:
                        persona_id, nombre, apellido, similitud = coincidencia
                        self.registrar_acceso(persona_id, similitud, frame)
                        
                        # Dibujar rectángulo y etiqueta
                        color = (0, 255, 0)  # Verde
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{nombre} {apellido} ({similitud:.2f})", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    else:
                        self.registrar_acceso(None, 0, frame)
                        color = (0, 0, 255)  # Rojo
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, "Desconocido", 
                                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                except Exception as e:
                    print(f"Error procesando rostro: {str(e)}")
        
        return frame

    def ejecutar(self):
        """Bucle principal de reconocimiento"""
        try:
            while True:
                ret, frame = self.camara.read()
                if not ret:
                    print("Error al capturar frame")
                    break
                
                frame_procesado = self.procesar_frame(frame)
                cv2.imshow('Reconocimiento Facial', frame_procesado)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.camara.release()
            cv2.destroyAllWindows()
            self.db_conn.close()

if __name__ == "__main__":
    reconocedor = ReconocimientoFacial()
    reconocedor.ejecutar()