FROM python:3.10-slim

WORKDIR /opt/is

# Install system-level dependencies required by OpenCV and MediaPipe
# Source - https://stackoverflow.com/a/63377623
# Posted by Tushar Kolhe, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-09, License - CC BY-SA 4.0

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY setup.py .
COPY etc/conf/options.json .

RUN pip install --no-cache-dir .

CMD ["is-skeletons-detector-stream"]
