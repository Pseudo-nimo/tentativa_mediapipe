import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

from is_msgs.image_pb2 import ObjectAnnotations, ObjectLabels, HumanKeypoints

from .utils import get_np_image
from .options_pb2 import SkeletonsDetectorOptions

# ---------------------------------------------------------------------------
# Model asset management
# ---------------------------------------------------------------------------
_MODEL_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'mediapipe')
_MODEL_NAMES = {
    0: 'pose_landmarker_lite.task',
    1: 'pose_landmarker_full.task',
    2: 'pose_landmarker_heavy.task',
}
_MODEL_URLS = {
    0: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
    1: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task',
    2: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',
}


def _get_model_path(complexity: int) -> str:
    """Return a local path to the MediaPipe Pose model file.

    Resolution order:
    1. Environment variable ``MEDIAPIPE_MODEL_PATH`` (path to any ``.task`` file).
    2. ``~/.cache/mediapipe/<model_name>.task`` if it already exists.
    3. Download from Google Storage and cache in ``~/.cache/mediapipe/``.
    """
    # 1) Explicit env-var override
    env_path = os.environ.get('MEDIAPIPE_MODEL_PATH', '')
    if env_path and os.path.isfile(env_path):
        return env_path

    complexity = max(0, min(2, complexity))
    os.makedirs(_MODEL_CACHE_DIR, exist_ok=True)
    cached = os.path.join(_MODEL_CACHE_DIR, _MODEL_NAMES[complexity])

    # 2) Already cached
    if os.path.isfile(cached):
        return cached

    # 3) Download
    url = _MODEL_URLS[complexity]
    print('Downloading MediaPipe Pose model ({}) from {} …'.format(
        _MODEL_NAMES[complexity], url))
    try:
        urllib.request.urlretrieve(url, cached)
        print('Model cached at {}'.format(cached))
    except Exception as exc:
        raise RuntimeError(
            'Could not download MediaPipe Pose model.\n'
            'Set the MEDIAPIPE_MODEL_PATH environment variable to point to a '
            'local .task file, or download manually from:\n  {}\n'
            'Original error: {}'.format(url, exc)
        ) from exc
    return cached


# ---------------------------------------------------------------------------
# Landmark → COCO-18 mapping
# ---------------------------------------------------------------------------
# MediaPipe Pose provides 33 landmarks (PoseLandmark enum).
# We expose the 17 that map directly to COCO-18 and approximate
# NECK as the midpoint between LEFT_SHOULDER and RIGHT_SHOULDER.
PL = mp.tasks.vision.PoseLandmark

_MP_TO_COCO = {
    PL.NOSE:           HumanKeypoints.Value('NOSE'),
    PL.LEFT_EYE:       HumanKeypoints.Value('LEFT_EYE'),
    PL.RIGHT_EYE:      HumanKeypoints.Value('RIGHT_EYE'),
    PL.LEFT_EAR:       HumanKeypoints.Value('LEFT_EAR'),
    PL.RIGHT_EAR:      HumanKeypoints.Value('RIGHT_EAR'),
    PL.LEFT_SHOULDER:  HumanKeypoints.Value('LEFT_SHOULDER'),
    PL.RIGHT_SHOULDER: HumanKeypoints.Value('RIGHT_SHOULDER'),
    PL.LEFT_ELBOW:     HumanKeypoints.Value('LEFT_ELBOW'),
    PL.RIGHT_ELBOW:    HumanKeypoints.Value('RIGHT_ELBOW'),
    PL.LEFT_WRIST:     HumanKeypoints.Value('LEFT_WRIST'),
    PL.RIGHT_WRIST:    HumanKeypoints.Value('RIGHT_WRIST'),
    PL.LEFT_HIP:       HumanKeypoints.Value('LEFT_HIP'),
    PL.RIGHT_HIP:      HumanKeypoints.Value('RIGHT_HIP'),
    PL.LEFT_KNEE:      HumanKeypoints.Value('LEFT_KNEE'),
    PL.RIGHT_KNEE:     HumanKeypoints.Value('RIGHT_KNEE'),
    PL.LEFT_ANKLE:     HumanKeypoints.Value('LEFT_ANKLE'),
    PL.RIGHT_ANKLE:    HumanKeypoints.Value('RIGHT_ANKLE'),
}

_NECK_ID = HumanKeypoints.Value('NECK')
RunningMode = mp.tasks.vision.RunningMode


class MediaPipePoseDetector:
    """Wraps MediaPipe Pose (Tasks API) and maps results to COCO-18 ObjectAnnotations."""

    def __init__(self, options: SkeletonsDetectorOptions):
        mp_cfg = options.mediapipe
        model_path = _get_model_path(mp_cfg.model_complexity)

        # static_image_mode=True  → IMAGE  (each frame processed independently)
        # static_image_mode=False → VIDEO  (landmark tracking across frames)
        self._static_mode = mp_cfg.static_image_mode
        running_mode = RunningMode.IMAGE if self._static_mode else RunningMode.VIDEO

        det_conf = mp_cfg.min_detection_confidence if mp_cfg.min_detection_confidence > 0.0 else 0.5
        trk_conf = mp_cfg.min_tracking_confidence if mp_cfg.min_tracking_confidence > 0.0 else 0.5

        po = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=10,
            min_pose_detection_confidence=det_conf,
            min_pose_presence_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        )
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(po)
        self._timestamp_ms = 0

        resize = options.resize
        self._resize_w = resize.width if resize.width > 0 else 0
        self._resize_h = resize.height if resize.height > 0 else 0

    def detect(self, image) -> ObjectAnnotations:
        """Run pose detection on *image* (np.ndarray BGR or is_msgs Image).

        Returns an ObjectAnnotations protobuf with COCO-18 keypoints.
        """
        frame = get_np_image(image)
        if frame is None or frame.size == 0:
            return ObjectAnnotations()

        if self._resize_w > 0 and self._resize_h > 0:
            frame = cv2.resize(frame, (self._resize_w, self._resize_h))

        im_h, im_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._static_mode:
            result = self._landmarker.detect(mp_image)
        else:
            self._timestamp_ms = int(time.monotonic() * 1000)
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        return self._to_object_annotations(result, im_w, im_h)

    def _to_object_annotations(self, result, im_w: int, im_h: int) -> ObjectAnnotations:
        obs = ObjectAnnotations()
        obs.resolution.width = im_w
        obs.resolution.height = im_h

        if not result.pose_landmarks:
            return obs

        for landmarks in result.pose_landmarks:
            ob = obs.objects.add()
            ob.label = 'human_skeleton'
            ob.id = ObjectLabels.Value('HUMAN_SKELETON')

            # Map directly available COCO-18 joints
            for mp_idx, coco_id in _MP_TO_COCO.items():
                lm = landmarks[mp_idx]
                part = ob.keypoints.add()
                part.id = coco_id
                part.position.x = lm.x * im_w
                part.position.y = lm.y * im_h
                part.score = lm.visibility if lm.visibility is not None else 0.0

            # Approximate NECK as midpoint between left and right shoulders
            ls = landmarks[PL.LEFT_SHOULDER]
            rs = landmarks[PL.RIGHT_SHOULDER]
            neck = ob.keypoints.add()
            neck.id = _NECK_ID
            neck.position.x = (ls.x + rs.x) / 2.0 * im_w
            neck.position.y = (ls.y + rs.y) / 2.0 * im_h
            ls_vis = ls.visibility if ls.visibility is not None else 0.0
            rs_vis = rs.visibility if rs.visibility is not None else 0.0
            neck.score = min(ls_vis, rs_vis)

        return obs

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
