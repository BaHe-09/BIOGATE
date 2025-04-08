import argparse
import cv2
import numpy as np
import psycopg2
from psycopg2.extras import Json  # Importación crítica añadida
from keras_facenet import FaceNet
from ultralytics import YOLO
import urllib.request
import os
from dotenv import load_dotenv
from datetime import datetime
import sys
import time

load_dotenv()

class FaceClassifier:
    def __init__(self):
        """Inicializa modelos y conexión a DB con manejo de errores mejorado"""
        try:
            print("⏳ Inicializando modelos...")
            start_time = time.time()
            
            # Carga modelos desde la carpeta models/
            self.yolo = YOLO('models/yolov8n-face.pt')  # Modelo específico para rostros
            self.facenet = FaceNet()
            
            # Conexión a Neon PostgreSQL
            self.db_conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
            
            print(f"✅ Modelos y DB inicializados en {(time.time()-start_time):.2f}s")
        except Exception as e:
            print(f"❌ Error en inicialización: {str(e)}")
            raise

    def descargar_imagen(self, url):
        """Descarga imagen desde URL con validación"""
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL debe comenzar con http:// o https://")
            
        temp_file = "/tmp/temp_image.jpg"
        try:
            urllib.request.urlretrieve(url, temp_file)
            if os.path.getsize(temp_file) == 0:
                raise ValueError("Imagen descargada está vacía")
            return temp_file
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise ValueError(f"Error al descargar imagen: {str(e)}")

    def extraer_rostro(self, image_path):
        """Extrae el rostro principal con YOLOv8-Face"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("No se pudo leer la imagen")
            
            start_time = time.time()
            results = self.yolo(img)
            inference_time = (time.time() - start_time) * 1000  # ms
            
            if not results or len(results[0].boxes) == 0:
                raise ValueError("No se detectaron rostros")
                
            # Obtener la caja con mayor confianza
            boxes = results[0].boxes
            main_box = boxes[np.argmax(boxes.conf.cpu().numpy())]
            x1, y1, x2, y2 = map(int, main_box.xyxy[0].cpu().numpy())
            
            # Validar y recortar rostro
            face = img[y1:y2, x1:x2]
            if face.size == 0:
                raise ValueError("Área del rostro inválida")
            
            print(f"👤 Rostro detectado en {inference_time:.1f}ms | Dimensión: {face.shape}")
            return face
            
        except Exception as e:
            raise ValueError(f"Error en detección facial: {str(e)}")

    def generar_embedding(self, face_image):
        """Genera embedding facial con FaceNet"""
        try:
            start_time = time.time()
            
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            embedding = self.facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
            
            if embedding.shape != (512,):
                raise ValueError("Dimensión de embedding incorrecta")
                
            print(f"🧠 Embedding generado en {(time.time()-start_time)*1000:.1f}ms")
            return embedding
            
        except Exception as e:
            raise ValueError(f"Error generando embedding: {str(e)}")

    def consultar_db(self, embedding, threshold=0.7):
        """Consulta la base de datos para coincidencias"""
        try:
            if not isinstance(embedding, np.ndarray) or embedding.shape != (512,):
                raise ValueError("Embedding debe ser numpy array de 512D")
                
            with self.db_conn.cursor() as cursor:
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
                
        except psycopg2.Error as e:
            print(f"❌ Error de base de datos: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Error inesperado en consulta: {str(e)}")
            return None

    def registrar_dispositivo(self):
        """Registra o obtiene dispositivo GitHub Camara"""
        try:
            with self.db_conn.cursor() as cursor:
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
                    '192.168.1.100',
                    'Activo'
                ))
                self.db_conn.commit()
                return cursor.fetchone()[0]
                
        except Exception as e:
            print(f"❌ Error al registrar dispositivo: {str(e)}")
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
                    Json({  # Usando el Json importado correctamente
                        "origen": "GitHub Actions",
                        "modelo": "YOLOv8-Face + FaceNet",
                        "timestamp": datetime.now().isoformat()
                    })
                ))
                self.db_conn.commit()
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"❌ Error al registrar acceso: {str(e)}")
            self.db_conn.rollback()
            return None

    def clasificar_rostro(self, image_url, threshold=0.7):
        """Flujo completo de clasificación con registro"""
        temp_path = None
        try:
            print(f"\n🔍 Iniciando procesamiento de imagen: {image_url}")
            
            # Paso 1: Descargar y procesar imagen
            temp_path = self.descargar_imagen(image_url)
            rostro = self.extraer_rostro(temp_path)
            embedding = self.generar_embedding(rostro)
            
            # Paso 2: Buscar en base de datos
            print(f"🔎 Buscando coincidencias (umbral: {threshold})...")
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
                print(f"  - ID Persona: {persona_id}")
                print(f"  - Nombre: {nombre} {apellido_p} {apellido_m or ''}")
                print(f"  - Email: {email}")
                print(f"  - Similitud: {similitud:.2%}")
                print(f"  - ID Registro Acceso: {acceso_id}")
                
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
            print("\n🏁 Proceso completado")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sistema de reconocimiento facial - Compara rostros con la base de datos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--image_url", 
        required=True,
        help="URL pública de la imagen a clasificar (ej: https://example.com/foto.jpg)"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.7,
        help="Umbral de similitud (0.5-0.9)"
    )
    
    args = parser.parse_args()
    
    try:
        classifier = FaceClassifier()
        resultado = classifier.clasificar_rostro(args.image_url, args.threshold)
        
        if resultado['status'] == 'success':
            print(f"\n✅ CLASIFICACIÓN EXITOSA")
            sys.exit(0)
        elif resultado['status'] == 'no_match':
            print(f"\n⚠️ COINCIDENCIA NO ENCONTRADA")
            sys.exit(2)
        else:
            print(f"\n❌ ERROR EN EL PROCESO")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Proceso interrumpido por el usuario")
        sys.exit(1)
