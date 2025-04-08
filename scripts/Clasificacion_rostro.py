import argparse
import cv2
import numpy as np
import psycopg2
from keras_facenet import FaceNet
from ultralytics import YOLO
import urllib.request
import os
from dotenv import load_dotenv
from datetime import datetime
import sys

load_dotenv()

class FaceClassifier:
    def __init__(self):
        """Inicializa modelos y conexión a DB"""
        try:
            # Modelos desde carpeta models/
            self.yolo = YOLO('models/yolov8n-face.pt')  # Usar modelo face específico
            self.facenet = FaceNet()
            
            # Conexión a Neon DB
            self.db_conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
            print("✅ Modelos y conexión a DB inicializados")
        except Exception as e:
            print(f"❌ Error en inicialización: {str(e)}")
            raise

    def descargar_imagen(self, url):
        """Descarga imagen desde URL"""
        temp_file = "/tmp/temp_image.jpg"
        urllib.request.urlretrieve(url, temp_file)
        return temp_file
        
    def extraer_rostro(self, image_path):
        """Extrae el rostro principal con YOLOv8-Face"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("No se pudo cargar la imagen")

        results = self.yolo(img)
        
        if not results or len(results[0].boxes) == 0:
            raise ValueError("No se detectaron rostros en la imagen")
            
        # Obtener la caja con mayor confianza
        boxes = results[0].boxes
        main_box = boxes[np.argmax(boxes.conf.cpu().numpy())]
        x1, y1, x2, y2 = map(int, main_box.xyxy[0].cpu().numpy())
        
        # Validar y recortar rostro
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            raise ValueError("El área del rostro es inválida")
            
        return face
        
    def generar_embedding(self, face_image):
        """Genera embedding facial con FaceNet"""
        face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        embedding = self.facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
        
        if embedding.shape != (512,):
            raise ValueError("Dimensión de embedding incorrecta")
            
        return embedding
        
    def consultar_db(self, embedding, threshold=0.7):
        """Consulta la base de datos para coincidencias"""
        try:
            if not isinstance(embedding, np.ndarray) or embedding.shape != (512,):
                raise ValueError("Embedding debe ser numpy array de 512D")
                
            with self.db_conn.cursor() as cursor:
                # Consulta mejorada con JOIN a personas
                cursor.execute("""
                    SELECT p.id_persona, p.nombre, p.apellido_paterno, 
                           p.apellido_materno, p.correo_electronico,
                           1 - (v.vector <=> %s::vector) as similitud
                    FROM vectores_identificacion v
                    JOIN personas p ON v.id_persona = p.id_persona
                    WHERE 1 - (v.vector <=> %s::vector) > %s
                    ORDER BY similitud DESC
                    LIMIT 1
                """, (embedding.tolist(), embedding.tolist(), float(threshold)))
                
                return cursor.fetchone()
                
        except Exception as e:
            print(f"Error en consulta SQL: {str(e)}")
            return None

    def registrar_dispositivo(self):
        """Registra o obtiene dispositivo GitHub Camara"""
        try:
            with self.db_conn.cursor() as cursor:
                # Buscar dispositivo existente
                cursor.execute("""
                    SELECT id_dispositivo FROM dispositivos 
                    WHERE nombre = 'GitHub Camara' LIMIT 1
                """)
                dispositivo = cursor.fetchone()
                
                if dispositivo:
                    return dispositivo[0]
                
                # Crear nuevo dispositivo si no existe
                cursor.execute("""
                    INSERT INTO dispositivos 
                    (nombre, tipo, ubicacion, direccion_ip, estado)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id_dispositivo
                """, (
                    'GitHub Camara',
                    'Cámara',
                    'Servidor GitHub Actions',
                    '192.168.1.100',  # IP de ejemplo
                    'Activo'
                ))
                self.db_conn.commit()
                return cursor.fetchone()[0]
                
        except Exception as e:
            print(f"Error al registrar dispositivo: {str(e)}")
            self.db_conn.rollback()
            return None

    def registrar_acceso(self, persona_id, dispositivo_id, confianza, foto_url="", resultado="Éxito"):
        """Registra intento de acceso en historial"""
        try:
            with self.db_conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO historial_accesos 
                    (id_persona, id_dispositivo, resultado, confianza, foto_url, metadatos)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id_acceso
                """, (
                    persona_id,
                    dispositivo_id,
                    resultado,
                    float(confianza),
                    foto_url,
                    Json({"origen": "GitHub Actions", "modelo": "YOLOv8-Face + FaceNet"})
                ))
                self.db_conn.commit()
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error al registrar acceso: {str(e)}")
            self.db_conn.rollback()
            return None

    def clasificar_rostro(self, image_url, threshold=0.7):
        """Flujo completo de clasificación con registro"""
        temp_path = None
        try:
            print(f"\n🔍 Procesando imagen: {image_url}")
            
            # Paso 1: Descargar y procesar imagen
            temp_path = self.descargar_imagen(image_url)
            rostro = self.extraer_rostro(temp_path)
            embedding = self.generar_embedding(rostro)
            
            # Paso 2: Buscar en base de datos
            coincidencia = self.consultar_db(embedding, threshold)
            
            # Paso 3: Registrar dispositivo
            dispositivo_id = self.registrar_dispositivo()
            if not dispositivo_id:
                raise ValueError("No se pudo obtener ID de dispositivo")
            
            # Paso 4: Registrar resultado
            if coincidencia:
                persona_id, nombre, apellido_p, apellido_m, email, similitud = coincidencia
                acceso_id = self.registrar_acceso(
                    persona_id=persona_id,
                    dispositivo_id=dispositivo_id,
                    confianza=similitud,
                    foto_url=image_url
                )
                
                print("\n🎯 Resultado de clasificación:")
                print(f"ID Persona: {persona_id}")
                print(f"Nombre: {nombre} {apellido_p} {apellido_m or ''}")
                print(f"Email: {email}")
                print(f"Similitud: {similitud:.2%}")
                print(f"ID Registro Acceso: {acceso_id}")
                
                return {
                    'status': 'success',
                    'persona_id': persona_id,
                    'similitud': similitud,
                    'acceso_id': acceso_id
                }
            else:
                acceso_id = self.registrar_acceso(
                    persona_id=None,
                    dispositivo_id=dispositivo_id,
                    confianza=0,
                    foto_url=image_url,
                    resultado="Fallo"
                )
                print("\n🔍 No se encontraron coincidencias por encima del umbral")
                return {
                    'status': 'no_match',
                    'acceso_id': acceso_id
                }
                
        except Exception as e:
            print(f"\n❌ Error durante clasificación: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clasifica un rostro comparando con la base de datos y registra el acceso"
    )
    parser.add_argument(
        "--image_url", 
        required=True,
        help="URL de la imagen a clasificar"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.7,
        help="Umbral de similitud (0.5-0.9)"
    )
    
    args = parser.parse_args()
    
    classifier = FaceClassifier()
    resultado = classifier.clasificar_rostro(args.image_url, args.threshold)
    
    if resultado['status'] == 'success':
        print(f"\n✅ Clasificación exitosa")
        sys.exit(0)
    elif resultado['status'] == 'no_match':
        print(f"\n⚠️ Coincidencia no encontrada")
        sys.exit(2)
    else:
        print(f"\n❌ Error en el proceso")
        sys.exit(1)
