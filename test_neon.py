import os
from dotenv import load_dotenv
import psycopg2
from psycopg2 import OperationalError, sql
from psycopg2.extras import Json
from datetime import datetime
import numpy as np

# Configuración
load_dotenv()
URL_BASE_DATOS = os.getenv('NEON_DATABASE_URL')
ID_PRUEBA = f"prueba_{datetime.now().strftime('%Y%m%d%H%M%S')}"

def generar_vector_prueba():
    """Genera un vector de prueba normalizado"""
    vector_aleatorio = np.random.rand(512)
    vector_normalizado = vector_aleatorio / np.linalg.norm(vector_aleatorio)
    return vector_normalizado.tolist()

def verificar_tablas(cursor):
    """Verifica la existencia de tablas esenciales"""
    tablas_requeridas = {'personas', 'historial_accesos', 'vectores_identificacion', 'dispositivos'}
    cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tablas_existentes = {row[0] for row in cursor.fetchall()}
    tablas_faltantes = tablas_requeridas - tablas_existentes
    
    if tablas_faltantes:
        print(f"Tablas faltantes: {', '.join(tablas_faltantes)}")
    else:
        print("Todas las tablas esenciales existen")
    
    return tablas_existentes

def probar_operaciones_vectores(conexion):
    """Prueba operaciones con vectores faciales"""
    try:
        with conexion.cursor() as cursor:
            # Crear persona de prueba
            cursor.execute("""
                INSERT INTO personas (nombre, apellido_paterno, correo_electronico)
                VALUES (%s, %s, %s)
                RETURNING id_persona
            """, ('Prueba Vector', 'Apellido', f'vector_{ID_PRUEBA}@gmail.com'))
            id_persona = cursor.fetchone()[0]
            
            # Insertar vector
            vector_prueba = generar_vector_prueba()
            cursor.execute("""
                INSERT INTO vectores_identificacion (id_persona, vector)
                VALUES (%s, %s)
            """, (id_persona, vector_prueba))
            
            # Búsqueda por similitud
            cursor.execute("""
                SELECT id_persona, vector <-> %s::vector AS distancia
                FROM vectores_identificacion
                ORDER BY distancia
                LIMIT 1
            """, (vector_prueba,))
            resultado = cursor.fetchone()
            print(f"Búsqueda vectorial: ID {resultado[0]} con distancia {resultado[1]:.4f}")
            
            conexion.commit()
            
            # Limpieza
            cursor.execute("DELETE FROM personas WHERE id_persona = %s", (id_persona,))
            conexion.commit()
            
    except Exception as error:
        conexion.rollback()
        print(f"Error en operaciones vectoriales: {error}")

def probar_historial_accesos(conexion, id_persona):
    """Prueba el registro en el historial de accesos"""
    try:
        with conexion.cursor() as cursor:
            # Obtener un dispositivo de prueba
            cursor.execute("SELECT id_dispositivo FROM dispositivos LIMIT 1")
            id_dispositivo = cursor.fetchone()[0] if cursor.rowcount > 0 else None
            
            if not id_dispositivo:
                cursor.execute("""
                    INSERT INTO dispositivos (nombre, tipo, ubicacion)
                    VALUES (%s, %s, %s)
                    RETURNING id_dispositivo
                """, ('Dispositivo Prueba', 'Cámara', 'Ubicación Prueba'))
                id_dispositivo = cursor.fetchone()[0]
                conexion.commit()
            
            # Registrar acceso
            cursor.execute("""
                INSERT INTO historial_accesos 
                (id_persona, id_dispositivo, resultado, confianza, metadatos)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id_acceso
            """, (id_persona, id_dispositivo, 'Éxito', 0.95, 
                 Json({'metodo': 'facial', 'prueba': True})))
            
            id_acceso = cursor.fetchone()[0]
            print(f"Registro de acceso creado (ID: {id_acceso})")
            
    except Exception as error:
        conexion.rollback()
        print(f"Error en historial de accesos: {error}")

def probar_operaciones_crud(conexion, tablas_existentes):
    """Prueba operaciones básicas CRUD"""
    if 'personas' not in tablas_existentes:
        return

    try:
        cursor = conexion.cursor()
        
        # CREATE
        cursor.execute("""
            INSERT INTO personas (nombre, apellido_paterno, correo_electronico, google_id)
            VALUES (%s, %s, %s, %s)
            RETURNING id_persona
        """, ('Persona Prueba', 'Prueba', f'{ID_PRUEBA}@gmail.com', ID_PRUEBA))
        id_persona = cursor.fetchone()[0]
        conexion.commit()
        print(f"Persona insertada (ID: {id_persona})")

        # READ
        cursor.execute("SELECT nombre FROM personas WHERE id_persona = %s", (id_persona,))
        resultado = cursor.fetchone()
        print(f"Registro leído: {resultado[0]}")

        # UPDATE
        cursor.execute("""
            UPDATE personas SET activo = FALSE 
            WHERE id_persona = %s
            RETURNING activo
        """, (id_persona,))
        actualizado = cursor.fetchone()[0]
        print(f"Registro actualizado (activo={actualizado})")

        # DELETE (limpieza)
        cursor.execute("DELETE FROM personas WHERE id_persona = %s", (id_persona,))
        conexion.commit()
        print("Datos de prueba eliminados")

    except Exception as error:
        conexion.rollback()
        print(f"Error en operaciones CRUD: {error}")

def ejecutar_pruebas():
    """Función principal para ejecutar todas las pruebas"""
    print("\n" + "="*50)
    print("Iniciando pruebas de base de datos Neon")
    print("="*50)
    
    try:
        with psycopg2.connect(URL_BASE_DATOS, sslmode="require") as conexion:
            print("\nConexión SSL establecida correctamente")
            
            with conexion.cursor() as cursor:
                # Verificar extensión vectorial
                cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
                print(f"\nExtensión vectorial: {'Activa' if cursor.fetchone()[0] else 'Faltante'}")
                
                # Verificar tablas
                print("\nVerificando estructura de la base de datos...")
                tablas_existentes = verificar_tablas(cursor)
                
                # Pruebas CRUD básicas
                if tablas_existentes:
                    print("\nProbando operaciones CRUD...")
                    probar_operaciones_crud(conexion, tablas_existentes)
                    
                    # Pruebas con vectores
                    if 'vectores_identificacion' in tablas_existentes:
                        print("\nProbando operaciones con vectores...")
                        probar_operaciones_vectores(conexion)
                    
                    # Pruebas con historial
                    if 'historial_accesos' in tablas_existentes and 'personas' in tablas_existentes:
                        print("\nProbando historial de accesos...")
                        with conexion.cursor() as cursor_temporal:
                            cursor_temporal.execute("""
                                INSERT INTO personas (nombre, apellido_paterno, correo_electronico)
                                VALUES (%s, %s, %s)
                                RETURNING id_persona
                            """, ('Prueba Historial', 'Test', f'historial_{ID_PRUEBA}@test.com'))
                            id_persona_prueba = cursor_temporal.fetchone()[0]
                            conexion.commit()
                            
                            probar_historial_accesos(conexion, id_persona_prueba)
                            
                            # Limpieza
                            cursor_temporal.execute("DELETE FROM personas WHERE id_persona = %s", (id_persona_prueba,))
                            conexion.commit()
                
    except OperationalError as error:
        print(f"\nError de conexión: {error}")
        print(f"¿Está correctamente configurada NEON_DATABASE_URL?")
        print(f"URL usada: {URL_BASE_DATOS[:30]}...")
    except Exception as error:
        print(f"\nError inesperado: {error}")
    finally:
        print("\n" + "="*50)
        print("Pruebas completadas")
        print("="*50 + "\n")

if __name__ == "__main__":
    ejecutar_pruebas()
