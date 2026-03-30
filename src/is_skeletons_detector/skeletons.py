from is_msgs.image_pb2 import ObjectAnnotations

from .mediapipe_pose import MediaPipePoseDetector
from .options_pb2 import SkeletonsDetectorOptions
from .utils import get_np_image


class SkeletonsDetector:
    """Public interface for skeleton detection.

    Wraps MediaPipePoseDetector and exposes the same ``detect()`` API
    as the original OpenPose-based implementation so that stream.py and
    rpc.py require no changes.
    """

    def __init__(self, options: SkeletonsDetectorOptions):
        if not isinstance(options, SkeletonsDetectorOptions):
            raise TypeError(
                "Invalid parameter on 'SkeletonsDetector' constructor: "
                "not a SkeletonsDetectorOptions type"
            )
        self._detector = MediaPipePoseDetector(options)

    def detect(self, image) -> ObjectAnnotations:
        """Detect human skeletons in *image*.

        Args:
            image: ``np.ndarray`` (BGR) or ``is_msgs.image_pb2.Image``.

        Returns:
            ``ObjectAnnotations`` with COCO-18 keypoints.
        """
        return self._detector.detect(image)

    def close(self):
        self._detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
