import argparse
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from ultralytics import YOLO
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

class FaceProcessor:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt')
        self.facenet = FaceNet()

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        results = self.yolo(img)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
        
        embeddings = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            
            if face.size == 0:
                continue
                
            try:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, (160, 160))
                embedding = self.facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error procesando imagen: {str(e)}")
        
        return embeddings

def register_person(full_name, folder_name):
    conn = None
    try:
        # Parsear nombre completo
        parts = full_name.split()
        nombre = parts[0]
        apellido_paterno = parts[1] if len(parts) > 1 else "Apellido"
        apellido_materno = parts[2] if len(parts) > 2 else None

        # Conectar a DB
        conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
        cursor = conn.cursor()
        
        # Registrar persona
        cursor.execute(
            """INSERT INTO personas 
            (nombre, apellido_paterno, apellido_materno, email) 
            VALUES (%s, %s, %s, %s)
            RETURNING id_persona""",
            (
                nombre,
                apellido_paterno,
                apellido_materno,
                f"{nombre.lower()}.{apellido_paterno.lower()}@example.com"
            )
        )
        person_id = cursor.fetchone()[0]
        
        # Procesar imágenes
        processor = FaceProcessor()
        registered = 0
        folder_path = os.path.join("dataset", folder_name)
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                embeddings = processor.process_image(img_path)
                
                if embeddings:
                    for emb in embeddings:
                        cursor.execute(
                            """INSERT INTO embeddings_faciales 
                            (id_persona, embedding, dispositivo_registro) 
                            VALUES (%s, %s, %s)""",
                            (person_id, emb.tolist(), 'GitHub Actions')
                        )
                    registered += len(embeddings)
        
        conn.commit()
        print(f"\n✅ Registro completado: {registered} embeddings para {full_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error en registro: {str(e)}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_name", required=True, help="Nombre completo con apellidos")
    parser.add_argument("--folder_name", required=True, help="Nombre de la carpeta en dataset/")
    args = parser.parse_args()
    
    print(f"\n=== INICIANDO REGISTRO PARA {args.full_name.upper()} ===")
    print(f"Buscando imágenes en: dataset/{args.folder_name}")
    
    if not register_person(args.full_name, args.folder_name):
        exit(1)
