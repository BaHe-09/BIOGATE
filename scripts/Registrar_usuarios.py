import argparse
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from ultralytics import YOLO
import psycopg2
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Optional, Tuple, Dict
import re

load_dotenv()

class ProcesadorFacial:
    """Clase para el procesamiento de rostros en imágenes"""
    
    def __init__(self):
        """Inicializa modelos YOLO y FaceNet desde carpeta models/"""
        self.modelo_deteccion = YOLO('models/yolov8n-face-lindevs.pt')  # Cambiado a carpeta models
        self.modelo_reconocimiento = FaceNet()
    
    def procesar_imagen(self, ruta_imagen: str) -> Optional[List[np.ndarray]]:
        """
        Detecta rostros y genera embeddings para una imagen
        
        Args:
            ruta_imagen: Path a la imagen a procesar
            
        Returns:
            Lista de embeddings faciales o None si hay error
        """
        try:
            img = cv2.imread(ruta_imagen)
            if img is None:
                print(f"Error: No se pudo leer la imagen {ruta_imagen}")
                return None
            
            # Detección de rostros
            resultados = self.modelo_deteccion(img)
            cajas = resultados[0].boxes.xyxy.cpu().numpy() if resultados else []
            
            embeddings = []
            for caja in cajas:
                x1, y1, x2, y2 = map(int, caja)
                rostro = img[y1:y2, x1:x2]
                
                if rostro.size == 0:
                    continue
                    
                try:
                    # Preprocesamiento para FaceNet
                    rostro_rgb = cv2.cvtColor(rostro, cv2.COLOR_BGR2RGB)
                    rostro_redimensionado = cv2.resize(rostro_rgb, (160, 160))
                    
                    # Generar embedding
                    embedding = self.modelo_reconocimiento.embeddings(
                        np.expand_dims(rostro_redimensionado, axis=0)
                    )[0]
                    embeddings.append(embedding)
                except Exception as e:
                    print(f"Error procesando rostro: {str(e)}")
            
            return embeddings
            
        except Exception as e:
            print(f"Error crítico en procesar_imagen: {str(e)}")
            return None

def validar_email(email: str) -> bool:
    """Valida el formato básico de un email"""
    return bool(re.match(r"[^@]+@[^@]+\.[^@]+", email)) if email else True

def registrar_persona_completa(
    nombre_completo: str, 
    nombre_carpeta: str,
    telefono: Optional[str] = None,
    email: Optional[str] = None
) -> Dict[str, int]:
    """
    Registra una nueva persona con todos sus datos en la base de datos
    
    Args:
        nombre_completo: Nombre completo (nombre apellido1 apellido2)
        nombre_carpeta: Nombre de la carpeta en dataset/ con las imágenes
        telefono: Número telefónico opcional
        email: Email opcional
        
    Returns:
        Dict con {
            'status': 0=éxito, 1=error,
            'id_persona': ID asignado,
            'embeddings': cantidad registrada
        }
    """
    resultado = {'status': 1, 'id_persona': None, 'embeddings': 0}
    conexion = None
    
    try:
        # Validación de inputs
        if not nombre_completo or not nombre_carpeta:
            print("Error: Nombre completo y carpeta son requeridos")
            return resultado
            
        if email and not validar_email(email):
            print("Error: Formato de email inválido")
            return resultado

        # Parsear nombre completo
        partes = [p.strip() for p in nombre_completo.split(maxsplit=2)]
        datos_persona = {
            'nombre': partes[0],
            'apellido_paterno': partes[1] if len(partes) > 1 else "",
            'apellido_materno': partes[2] if len(partes) > 2 else None,
            'telefono': telefono,
            'email': email or f"{partes[0].lower()}.{partes[1].lower() if len(partes) > 1 else 'user'}@example.com"
        }

        # Verificar carpeta de imágenes
        ruta_carpeta = os.path.join("dataset", nombre_carpeta)
        if not os.path.exists(ruta_carpeta):
            print(f"Error: Carpeta {ruta_carpeta} no existe")
            return resultado

        # Conectar a DB
        conexion = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
        cursor = conexion.cursor()
        
        # Registrar persona con todos los campos
        cursor.execute(
            """INSERT INTO personas 
            (nombre, apellido_paterno, apellido_materno, telefono, correo_electronico, activo) 
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id_persona""",
            (
                datos_persona['nombre'],
                datos_persona['apellido_paterno'],
                datos_persona['apellido_materno'],
                datos_persona['telefono'],
                datos_persona['email'],
                True  # Siempre activo al registrar
            )
        )
        id_persona = cursor.fetchone()[0]
        resultado['id_persona'] = id_persona
        
        # Procesar imágenes
        procesador = ProcesadorFacial()
        archivos_procesados = 0
        embeddings_registrados = 0
        
        for archivo in os.listdir(ruta_carpeta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_imagen = os.path.join(ruta_carpeta, archivo)
                embeddings = procesador.procesar_imagen(ruta_imagen)
                archivos_procesados += 1
                
                if embeddings:
                    for emb in embeddings:
                        cursor.execute(
                            """INSERT INTO vectores_identificacion 
                            (id_persona, vector, dispositivo_registro, modelo) 
                            VALUES (%s, %s, %s, %s)""",
                            (id_persona, emb.tolist(), 'GitHub Actions', 'Facenet')
                        )
                    embeddings_registrados += len(embeddings)
        
        conexion.commit()
        resultado['embeddings'] = embeddings_registrados
        resultado['status'] = 0
        
        print(f"\n✅ Registro completado - ID: {id_persona}")
        print(f"📊 Resumen:")
        print(f"  - Nombre: {datos_persona['nombre']} {datos_persona['apellido_paterno']}")
        print(f"  - Email: {datos_persona['email']}")
        print(f"  - Teléfono: {datos_persona['telefono'] or 'No proporcionado'}")
        print(f"  - Imágenes procesadas: {archivos_procesados}")
        print(f"  - Embeddings registrados: {embeddings_registrados}")
        
    except psycopg2.Error as e:
        print(f"❌ Error de base de datos: {str(e)}")
        if conexion:
            conexion.rollback()
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")
        if conexion:
            conexion.rollback()
    finally:
        if conexion:
            conexion.close()
    
    return resultado

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Registra una nueva persona con todos sus datos y embeddings faciales")
    
    parser.add_argument(
        "--full_name", 
        required=True, 
        help="Nombre completo con formato 'Nombre Apellido1 Apellido2'")
    
    parser.add_argument(
        "--folder_name", 
        required=True, 
        help="Nombre de la carpeta dentro de dataset/ que contiene las imágenes")
    
    parser.add_argument(
        "--phone",
        required=False,
        default=None,
        help="Número de teléfono (opcional)")
    
    parser.add_argument(
        "--email",
        required=False,
        default=None,
        help="Email (opcional)")
    
    args = parser.parse_args()
    
    print(f"\n=== INICIANDO REGISTRO COMPLETO ===")
    print(f"👤 Nombre: {args.full_name}")
    print(f"📁 Carpeta: dataset/{args.folder_name}")
    if args.phone:
        print(f"📞 Teléfono: {args.phone}")
    if args.email:
        print(f"✉️ Email: {args.email}")
    
    resultado = registrar_persona_completa(
        args.full_name,
        args.folder_name,
        args.phone,
        args.email
    )
    
    if resultado['status'] != 0:
        print("\n❌ El registro no se completó correctamente")
        exit(1)
    
    exit(0)
    
    print(f"\n=== INICIANDO REGISTRO PARA {args.full_name.upper()} ===")
    print(f"Buscando imágenes en: dataset/{args.folder_name}")
    
    if not register_person(args.full_name, args.folder_name):
        exit(1)
