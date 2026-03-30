# is-skeletons-detector (MediaPipe Pose)

Skeleton detection service based on **MediaPipe Pose**, built for the
[Intelligent Spaces (IS)](https://github.com/labviros) ecosystem.  
It replaces the original OpenPose/tf-pose-estimation back-end with
[MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
while keeping the same AMQP stream and RPC interfaces.

---

## Streams

| Name | Input (Topic / Message) | Output (Topic / Message) | Description |
|------|------------------------|--------------------------|-------------|
| Skeletons.Detection | **CameraGateway.\*.Frame** `[Image]` | **SkeletonsDetector.\*.Detection** `[ObjectAnnotations]` | Detect human skeletons and publish keypoints |
| Skeletons.Rendered  | **CameraGateway.\*.Frame** `[Image]` | **SkeletonsDetector.\*.Rendered**  `[Image]` | Skeleton overlay drawn on the input image |

Run with:

```bash
is-skeletons-detector-stream
```

---

## RPCs

| Service | Request | Reply | Description |
|---------|---------|-------|-------------|
| SkeletonsDetector.Detect | `[Image]` | `[ObjectAnnotations]` | Single-shot skeleton detection |

Run with:

```bash
is-skeletons-detector-rpc
```

---

## Keypoints

The service outputs **COCO-18** keypoints using the
[`HumanKeypoints`](https://github.com/labviros/is-msgs/blob/modern-cmake/docs/README.md#humankeypoints)
enum from `is-msgs`.

| ID | Joint |
|----|-------|
| NOSE | Nose |
| NECK | Mid-point between left & right shoulders (approximated) |
| LEFT_SHOULDER / RIGHT_SHOULDER | Shoulders |
| LEFT_ELBOW / RIGHT_ELBOW | Elbows |
| LEFT_WRIST / RIGHT_WRIST | Wrists |
| LEFT_HIP / RIGHT_HIP | Hips |
| LEFT_KNEE / RIGHT_KNEE | Knees |
| LEFT_ANKLE / RIGHT_ANKLE | Ankles |
| LEFT_EYE / RIGHT_EYE | Eyes |
| LEFT_EAR / RIGHT_EAR | Ears |

MediaPipe Pose provides 33 landmarks.  
The service maps the 17 directly available COCO joints plus approximates
`NECK` as the midpoint between `LEFT_SHOULDER` and `RIGHT_SHOULDER`.

---

## Configuration

Edit `etc/conf/options.json`:

```json
{
  "broker_uri": "amqp://localhost:5672",
  "zipkin_host": "localhost",
  "zipkin_port": 9411,
  "zipkin_batch_size": 100,
  "mediapipe": {
    "static_image_mode": false,
    "model_complexity": 1,
    "smooth_landmarks": true,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5
  },
  "resize": {
    "width": 0,
    "height": 0
  }
}
```

Pass a custom path as the first CLI argument:

```bash
is-skeletons-detector-stream /path/to/my-options.json
```

### MediaPipe options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `static_image_mode` | bool | `false` | Treat every frame as independent (no tracking) |
| `model_complexity` | int | `1` | Model complexity: 0 (lite), 1 (full), 2 (heavy) |
| `smooth_landmarks` | bool | `true` | Smooth landmarks across frames |
| `min_detection_confidence` | float | `0.5` | Minimum detection confidence |
| `min_tracking_confidence` | float | `0.5` | Minimum tracking confidence |

### Resize (optional)

Set `resize.width` and `resize.height` to a non-zero value to scale
images before inference (can speed things up on high-resolution streams).
Set both to `0` to disable resizing.

---

## Installation

```bash
pip install -e .
```

Dependencies installed automatically:

- `is-wire`
- `is-msgs`
- `mediapipe`
- `opencv-python`
- `python-dateutil`

---

## Docker

### Build and run (CPU)

```bash
docker build -f etc/docker/Dockerfile -t is-skeletons-detector .
docker run --rm --network=host \
    -v $(pwd)/etc/conf/options.json:/opt/is/options.json \
    is-skeletons-detector is-skeletons-detector-stream
```

### Development image

```bash
docker build -f etc/docker/Dockerfile.dev -t is-skeletons-detector/dev .
docker run -ti --rm --network=host -v $(pwd):/devel is-skeletons-detector/dev bash
```

---

## Protobuf options

If you modify `src/conf/options.proto`, regenerate the Python module:

```bash
pip install grpcio-tools
python -m grpc_tools.protoc \
    -I src/conf \
    --python_out=src/is_skeletons_detector \
    src/conf/options.proto
```
