FROM python:3.10-slim

WORKDIR /opt/is

# Install system-level dependencies required by OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY setup.py .
COPY etc/conf/options.json .

RUN pip install --no-cache-dir .

CMD ["is-skeletons-detector-stream"]
