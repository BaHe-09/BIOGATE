import argparse
import os
import psycopg2
from dotenv import load_dotenv
from datetime import datetime

def register_person(full_name):
    try:
        # Parsear nombre completo
        name_parts = full_name.split()
        nombre = name_parts[0]
        apellido_paterno = name_parts[1] if len(name_parts) > 1 else "Desconocido"
        apellido_materno = name_parts[2] if len(name_parts) > 2 else None
        
        conn = psycopg2.connect(os.getenv('NEON_DATABASE_URL'))
        cursor = conn.cursor()
        
        # Insertar persona con todos los campos requeridos
        cursor.execute(
            """INSERT INTO personas 
            (nombre, apellido_paterno, apellido_materno, email, google_id) 
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id_persona""",
            (
                nombre,
                apellido_paterno,
                apellido_materno,
                f"{nombre.lower()}_{datetime.now().strftime('%Y%m%d')}@test.com",
                f"{nombre.lower()}_testid"
            )
        )
        
        person_id = cursor.fetchone()[0]
        conn.commit()
        return person_id
        
    except Exception as e:
        print(f"Error registrando persona: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_name", required=True, help="Nombre completo incluyendo apellidos")
    args = parser.parse_args()
    
    load_dotenv()
    print(f"=== REGISTRO PARA {args.full_name.upper()} ===")
    
    try:
        person_id = register_person(args.full_name)
        print(f"✅ Persona registrada con ID: {person_id}")
        
        # Resto de tu lógica para procesar imágenes...
        
    except Exception as e:
        print(f"❌ Error en el proceso: {str(e)}")
        exit(1)
