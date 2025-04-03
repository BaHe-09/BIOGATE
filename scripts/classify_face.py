import argparse
import cv2
import numpy as np
import psycopg2
from dotenv import load_dotenv
from keras_facenet import FaceNet
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import os

load_dotenv()

class FaceClassifier:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt')
        self.facenet = FaceNet()
        self.db_conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
        
    def download_image(self, url):
        """Descarga imagen desde URL"""
        temp_file = "/tmp/temp_image.jpg"
        urllib.request.urlretrieve(url, temp_file)
        return temp_file
        
    def extract_face(self, image_path):
        """Extrae el rostro principal de una imagen"""
        img = cv2.imread(image_path)
        results = self.yolo(img)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
        
        if len(boxes) == 0:
            raise ValueError("No se detectaron rostros en la imagen")
            
        # Tomar el rostro con mayor confianza
        main_box = sorted(boxes, key=lambda x: x[4], reverse=True)[0]
        x1, y1, x2, y2 = map(int, main_box[:4])
        face = img[y1:y2, x1:x2]
        
        # Preprocesamiento para FaceNet
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        return face_resized
        
    def get_embedding(self, face_image):
        """Genera embedding facial"""
        return self.facenet.embeddings(np.expand_dims(face_image, axis=0))[0]
        
    def query_database(self, embedding, threshold):
        """Consulta la base de datos por coincidencias"""
        with self.db_conn.cursor() as cursor:
            cursor.execute("""
                SELECT p.id_persona, p.nombre, p.apellido_paterno, 
                       e.embedding, 
                       1 - (e.embedding <=> %s) as similitud
                FROM embeddings_faciales e
                JOIN personas p ON e.id_persona = p.id_persona
                WHERE 1 - (e.embedding <=> %s) > %s
                ORDER BY similitud DESC
                LIMIT 5
            """, (embedding.tolist(), embedding.tolist(), threshold))
            return cursor.fetchall()
            
    def classify(self, image_url, threshold=0.75):
        """Flujo completo de clasificación"""
        try:
            print(f"\nClasificando imagen: {image_url}")
            temp_path = self.download_image(image_url)
            face = self.extract_face(temp_path)
            embedding = self.get_embedding(face)
            
            matches = self.query_database(embedding, threshold)
            
            if not matches:
                print("\n🔍 No se encontraron coincidencias por encima del umbral")
                return None
                
            print("\n🎯 Resultados de clasificación:")
            for i, match in enumerate(matches, 1):
                person_id, nombre, apellido, _, similitud = match
                print(f"{i}. {nombre} {apellido} - Similitud: {similitud:.2f}")
                
            best_match = matches[0]
            return best_match
            
        except Exception as e:
            print(f"\n❌ Error durante clasificación: {str(e)}")
            return None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.db_conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", required=True, help="URL de la imagen a clasificar")
    parser.add_argument("--threshold", type=float, default=0.75, help="Umbral de similitud (0-1)")
    args = parser.parse_args()
    
    classifier = FaceClassifier()
    result = classifier.classify(args.image_url, args.threshold)
    
    if result:
        person_id, nombre, apellido, _, similitud = result
        print(f"\n✅ Mejor coincidencia: {nombre} {apellido} (ID: {person_id}) con similitud {similitud:.2f}")
