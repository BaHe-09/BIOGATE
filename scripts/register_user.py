import os
import cv2
import numpy as np
import argparse
import psycopg2
from dotenv import load_dotenv
from keras_facenet import FaceNet
from ultralytics import YOLO
from tqdm import tqdm

# Configuración
load_dotenv()
DATABASE_URL = os.getenv('NEON_DATABASE_URL')
MODEL_YOLO_PATH = 'models/yolov8n-face-lindevs.pt'
DATASET_PATH = 'dataset'
FACE_SIZE = (160, 160)
MIN_FACE_SIZE = 30

class FaceEmbedder:
    def __init__(self):
        self.yolo = YOLO(MODEL_YOLO_PATH)
        self.facenet = FaceNet()
    
    def process_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Detección de rostros
        results = self.yolo(img, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
        
        embeddings = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = img[y1:y2, x1:x2]
            
            if face.shape[0] < MIN_FACE_SIZE or face.shape[1] < MIN_FACE_SIZE:
                continue
                
            try:
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_resized = cv2.resize(face_rgb, FACE_SIZE)
                embedding = self.facenet.embeddings(np.expand_dims(face_resized, axis=0))[0]
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error procesando {img_path}: {str(e)}")
        
        return embeddings

def register_person(person_name):
    # 1. Registrar persona en la base de datos
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Insertar persona
        cursor.execute(
            "INSERT INTO personas (nombre, email) VALUES (%s, %s) RETURNING id_persona",
            (person_name, f"{person_name}@test.com")
        )
        person_id = cursor.fetchone()[0]
        conn.commit()
        
        # 2. Procesar imágenes
        embedder = FaceEmbedder()
        person_folder = os.path.join(DATASET_PATH, person_name)
        registered = 0
        
        for filename in tqdm(os.listdir(person_folder), desc="Procesando imágenes"):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_folder, filename)
                embeddings = embedder.process_image(img_path)
                
                if embeddings:
                    for emb in embeddings:
                        cursor.execute(
                            "INSERT INTO embeddings_faciales (id_persona, embedding, dispositivo_registro) VALUES (%s, %s, %s)",
                            (person_id, emb.tolist(), 'GitHub_Actions')
                        )
                    registered += len(embeddings)
        
        conn.commit()
        print(f"\n✅ Registro completado: {registered} embeddings para {person_name} (ID: {person_id})")
        
    except Exception as e:
        print(f"❌ Error en registro: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person_name", required=True, help="Nombre de la persona a registrar")
    args = parser.parse_args()
    
    print(f"=== INICIANDO REGISTRO PARA {args.person_name.upper()} ===")
    register_person(args.person_name)