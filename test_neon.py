import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError, sql
from psycopg2.extras import Json

# Cargar variables de entorno
load_dotenv()

# Configuración
DATABASE_URL = os.getenv("NEON_DATABASE_URL")
TEST_EMAIL = "test_github_actions@example.com"

def test_connection():
    conn = None
    try:
        # 1. Conexión con SSL
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        cursor = conn.cursor()
        print("✅ Conexión exitosa a Neon!")

        # 2. Verificar extensión vectorial
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');")
        print(f"🔹 Extensión vectorial: {'✅ Activa' if cursor.fetchone()[0] else '❌ Inexistente'}")

        # 3. Verificar tablas principales
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('personas', 'accesos', 'embeddings_faciales');
        """)
        existing_tables = [row[0] for row in cursor.fetchall()]
        print("🔍 Tablas encontradas:", ", ".join(existing_tables) or "Ninguna")

        # 4. Prueba CRUD en 'personas' (si existe la tabla)
        if 'personas' in existing_tables:
            # Insertar datos de prueba
            cursor.execute("""
                INSERT INTO personas (nombre, apellido_paterno, email)
                VALUES (%s, %s, %s)
                RETURNING id_persona;
            """, ("Test", "GitHub", TEST_EMAIL))
            persona_id = cursor.fetchone()[0]
            conn.commit()
            print(f"🔹 Insertado registro prueba en 'personas' (ID: {persona_id})")

            # Verificar inserción
            cursor.execute("SELECT nombre FROM personas WHERE id_persona = %s;", (persona_id,))
            print(f"📝 Registro verificado: {cursor.fetchone()[0]}")

        # 5. Limpieza (eliminar datos de prueba)
        if 'personas' in existing_tables:
            cursor.execute("DELETE FROM personas WHERE email = %s;", (TEST_EMAIL,))
            conn.commit()
            print("🧹 Datos de prueba eliminados")

    except OperationalError as e:
        print(f"❌ Error de conexión: {e}")
    except Exception as e:
        print(f"⚠️ Error inesperado: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("🚀 Iniciando prueba de conexión a Neon...")
    test_connection()
    print("🏁 Prueba completada")