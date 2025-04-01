import os
import cv2
import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2 import sql
from datetime import datetime
from ultralytics import YOLO
from keras_facenet import FaceNet

# Configuración
load_dotenv()
DATABASE_URL = os.getenv('NEON_DATABASE_URL')
TEST_ID = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"

# Inicializar modelos
modelo_yolo = YOLO('yolov8n-face-lindevs.pt')
facenet = FaceNet()

def create_db_connection():
    """Establece conexión a la base de datos Neon"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        print(f"Error de conexión a la DB: {e}")
        return None

def register_test_user(conn):
    """Registra un usuario de prueba en la base de datos"""
    try:
        with conn.cursor() as cursor:
            # Registrar persona
            cursor.execute(
                sql.SQL("""
                INSERT INTO personas 
                (nombre, apellido_paterno, email, google_id) 
                VALUES (%s, %s, %s, %s)
                RETURNING id_persona
                """),
                ('Test_User', 'Auto_Register', f'{TEST_ID}@test.com', f'{TEST_ID}_google')
            )
            id_persona = cursor.fetchone()[0]
            
            # Registrar cuenta asociada
            cursor.execute(
                sql.SQL("""
                INSERT INTO cuentas 
                (id_persona, id_rol, nombre_usuario, contraseña_hash, salt) 
                VALUES (%s, 2, %s, 'test_hash', 'test_salt')
                """),
                (id_persona, f'user_{TEST_ID}')
            )
            
            conn.commit()
            return id_persona
    except Exception as e:
        print(f"Error registrando usuario: {e}")
        conn.rollback()
        return None

def capture_and_register_embeddings(conn, id_persona, num_samples=5):
    """Captura rostros desde cámara y registra embeddings"""
    cap = cv2.VideoCapture(0)
    registered = 0
    
    print(f"\nCapturando {num_samples} muestras para el usuario {id_persona}...")
    print("Presione 's' para guardar el rostro detectado, 'q' para terminar")
    
    while registered < num_samples:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Detección con YOLO
        resultados = modelo_yolo(frame, verbose=False)
        boxes = resultados[0].boxes.xyxy.cpu().numpy() if resultados else []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            rostro = frame[y1:y2, x1:x2]
            
            if rostro.size == 0:
                continue
                
            # Mostrar preview
            preview = frame.copy()
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(preview, f"Muestra {registered+1}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Registro Facial - Presione 's' para guardar", preview)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                # Generar embedding
                rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
                rostro_resized = cv2.resize(rostro_rgb, (160, 160))
                embedding = facenet.embeddings(np.expand_dims(rostro_resized, axis=0))
                
                # Registrar en DB
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            sql.SQL("""
                            INSERT INTO embeddings_faciales 
                            (id_persona, embedding, dispositivo_registro) 
                            VALUES (%s, %s, %s)
                            """),
                            (id_persona, embedding[0].tolist(), 'GitHub_Actions')
                        )
                        conn.commit()
                        registered += 1
                        print(f"✅ Embedding {registered} registrado")
                except Exception as e:
                    print(f"Error registrando embedding: {e}")
                    conn.rollback()
            
            elif key == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()
    return registered

if __name__ == "__main__":
    print("=== REGISTRO AUTOMÁTICO DE USUARIO DE PRUEBA ===")
    
    conn = create_db_connection()
    if conn:
        id_persona = register_test_user(conn)
        if id_persona:
            print(f"\nUsuario de prueba creado con ID: {id_persona}")
            samples_registered = capture_and_register_embeddings(conn, id_persona)
            print(f"\nResumen: {samples_registered} embeddings registrados para usuario {id_persona}")
        conn.close()
