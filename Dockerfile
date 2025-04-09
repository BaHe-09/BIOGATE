# Usa una imagen base más ligera con las dependencias necesarias
FROM python:3.10-slim

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Configura el entorno
WORKDIR /app

# Instala dependencias de Python en dos pasos (para mejor caché)
COPY requirements.txt .

# Primero instala las dependencias más ligeras
RUN pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    python-dotenv \
    psycopg2-binary \
    tqdm \
    scikit-learn

# Luego instala TensorFlow con timeout extendido
RUN pip install --no-cache-dir --timeout=100 \
    tensorflow-cpu==2.12.0 \
    keras-facenet \
    ultralytics \
    torch

# Copia el resto de la aplicación
COPY . .

# Descarga el modelo YOLO
RUN mkdir -p models && \
    wget https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt -O models/yolov8n-face.pt

# Variables de entorno
ENV NEON_DATABASE_URL="tu_url_de_conexion"
ENV DISPLAY=:0

# Comando de ejecución
CMD ["python", "camara_reconocimiento.py"]
