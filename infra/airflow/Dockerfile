FROM apache/airflow:3.1.1rc1-python3.11

# Install system dependencies for OpenCV
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy custom ultralytics-kd package (build context is project root)
COPY --chown=airflow:root training_pipeline/src/ultralytics-kd /tmp/ultralytics-kd

# Copy data_validation module for drift detection
COPY --chown=airflow:root data_validation /opt/airflow/data_validation

# Copy and setup dependencies
COPY infra/airflow/requirements.txt /opt/airflow/requirements.txt

# Install ultralytics-kd first (custom fork with KD support)
RUN pip install --no-cache-dir /tmp/ultralytics-kd

# Install other requirements (excluding standard ultralytics packages)
RUN grep -v "^ultralytics" /opt/airflow/requirements.txt > /tmp/requirements-filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements-filtered.txt

# Clean up
RUN rm -rf /tmp/ultralytics-kd /tmp/requirements-filtered.txt

# Set working directory
WORKDIR /opt/airflow