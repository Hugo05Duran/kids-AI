# Utilizar una imagen base oficial de Python
FROM python:3.9-slim

# Establecer la variable de entorno para no generar archivos .pyc
ENV PYTHONUNBUFFERED=1

# Establecer el directorio de trabajo
WORKDIR /app

# Instalar las dependencias del sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libc-dev \
        libpq-dev \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar los archivos de requisitos del proyecto
COPY requirements.txt /app/

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación
COPY . /app/

