import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError, sql
from psycopg2.extras import Json
from datetime import datetime

# Configuración
load_dotenv()  # Carga .env solo en desarrollo local
DATABASE_URL = os.getenv('NEON_DATABASE_URL')
TEST_ID = f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}"

def verify_tables(cursor):
    """Verifica la existencia de tablas esenciales"""
    required_tables = {'personas', 'accesos', 'embeddings_faciales'}
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    existing_tables = {row[0] for row in cursor.fetchall()}
    missing_tables = required_tables - existing_tables
    
    if missing_tables:
        print(f"⚠️ Tablas faltantes: {', '.join(missing_tables)}")
    else:
        print("✅ Todas las tablas esenciales existen")
    
    return existing_tables

def test_crud_operations(conn, existing_tables):
    """Prueba operaciones básicas CRUD"""
    if 'personas' not in existing_tables:
        return

    try:
        cursor = conn.cursor()
        
        # CREATE
        cursor.execute("""
            INSERT INTO personas (nombre, apellido_paterno, email, google_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id_persona
        """, ('Test', 'GitHub', f'{TEST_ID}@example.com', TEST_ID))
        persona_id = cursor.fetchone()[0]
        conn.commit()
        print(f"🔹 Persona insertada (ID: {persona_id})")

        # READ
        cursor.execute("SELECT nombre FROM personas WHERE id_persona = %s", (persona_id,))
        result = cursor.fetchone()
        print(f"Registro leído: {result[0]}")

        # UPDATE
        cursor.execute("""
            UPDATE personas SET activo = FALSE 
            WHERE id_persona = %s
            RETURNING activo
        """, (persona_id,))
        updated = cursor.fetchone()[0]
        print(f"Registro actualizado (activo={updated})")

        # DELETE (limpieza)
        cursor.execute("DELETE FROM personas WHERE id_persona = %s", (persona_id,))
        conn.commit()
        print("Datos de prueba eliminados")

    except Exception as e:
        conn.rollback()
        print(f"Error en operaciones CRUD: {e}")

def test_connection():
    """Prueba principal de conexión y funcionalidad"""
    print("\n" + "="*50)
    print("Iniciando pruebas de Neon DB")
    print("="*50)
    
    conn = None
    try:
        # 1. Conexión SSL
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        cursor = conn.cursor()
        print("\n--- Conexión SSL establecida correctamente ---")

        # 2. Verificar extensión vectorial
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        print(f"\nExtensión vectorial: {'✅ Activa' if cursor.fetchone()[0] else '❌ Faltante'}")

        # 3. Verificar tablas
        print("\nVerificando estructura de la base de datos...")
        existing_tables = verify_tables(cursor)

        # 4. Pruebas CRUD
        if existing_tables:
            print("\nProbando operaciones CRUD...")
            test_crud_operations(conn, existing_tables)

        # 5. Verificar índices (opcional)
        if 'embeddings_faciales' in existing_tables:
            cursor.execute("""
                SELECT indexname FROM pg_indexes 
                WHERE tablename = 'embeddings_faciales'
                AND indexdef LIKE '%ivfflat%'
            """)
            print(f"\n🔎 Índice vectorial encontrado: {'✅ Sí' if cursor.fetchone() else '❌ No'}")

    except OperationalError as e:
        print(f"\n❌ Error de conexión: {e}")
        print(f"ℹ️ ¿Estás seguro que NEON_DATABASE_URL está correctamente configurada?")
        print(f"ℹ️ URL usada: {DATABASE_URL[:30]}...")  # Muestra solo parte inicial por seguridad
    except Exception as e:
        print(f"\n⚠️ Error inesperado: {e}")
    finally:
        if conn:
            conn.close()
        print("\n" + "="*50)
        print("🏁 Pruebas completadas")
        print("="*50 + "\n")

if __name__ == "__main__":
    test_connection()
