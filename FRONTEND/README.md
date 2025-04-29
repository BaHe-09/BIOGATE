Estado actual del sistema BioGate – Conexión frontend con backend
El proyecto BioGate ya cuenta con dos componentes clave completamente desarrollados:

Frontend móvil desarrollado con React Native utilizando expo-router, donde ya se han implementado todas las pantallas principales: inicio de sesión, historial de accesos, detalle de usuario, registro de nuevo usuario, y configuración del sistema.

Backend desplegado en la plataforma Render, con una base de datos PostgreSQL alojada en Neon, el cual expone rutas API para realizar operaciones como login, registro, consulta y gestión de accesos.

Actualmente, la única etapa pendiente es establecer la conexión entre el frontend y el backend, de forma que los datos utilizados en la aplicación móvil ya no estén simulados o escritos directamente en el código, sino que provengan de la base de datos real mediante peticiones HTTP.

Objetivos de esta fase
Validar el inicio de sesión de los usuarios utilizando datos reales desde la base de datos.

Consultar y mostrar en la aplicación los registros reales de accesos.

Registrar nuevos usuarios y guardar esa información de manera persistente en la base de datos.

Mostrar datos individuales de un usuario al seleccionarlo, directamente desde la fuente real.

Preparar el sistema para futuras operaciones como edición o eliminación de registros desde la app.

Pasos necesarios
Establecer la dirección del backend como punto de entrada para todas las peticiones desde el frontend.

Reemplazar los datos simulados (por ejemplo, accesos o usuarios definidos manualmente) por datos obtenidos desde el backend mediante solicitudes GET y POST.

Enviar los formularios desde la app (como el de inicio de sesión o registro) al backend utilizando el formato requerido por las rutas existentes.

Manejar correctamente la respuesta del servidor, incluyendo casos de éxito, error, validaciones y mensajes informativos para el usuario.

Agregar indicadores de carga o mensajes de espera mientras se procesan las peticiones, para mejorar la experiencia de usuario.

Validar la seguridad y consistencia de los datos, asegurando que la aplicación reaccione adecuadamente ante errores de red o problemas del servidor.


