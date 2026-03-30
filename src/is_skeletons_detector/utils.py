import sys
import cv2
import numpy as np
from itertools import permutations
from google.protobuf.json_format import Parse

from is_wire.core import Logger
from is_msgs.image_pb2 import Image, ObjectAnnotations
from is_msgs.image_pb2 import HumanKeypoints as HKP

from .options_pb2 import SkeletonsDetectorOptions


def load_options() -> SkeletonsDetectorOptions:
    log = Logger(name='LoadOptions')
    op_file = sys.argv[1] if len(sys.argv) > 1 else 'options.json'
    try:
        with open(op_file, 'r') as f:
            try:
                op = Parse(f.read(), SkeletonsDetectorOptions())
                log.info('SkeletonsDetectorOptions: \n{}', op)
                return op
            except Exception as ex:
                log.critical("Unable to load options from '{}'. \n{}", op_file, ex)
    except Exception as ex:
        log.critical("Unable to open file '{}'", op_file)
    # Logger.critical already calls exit(-1); this line is never reached.
    raise SystemExit(1)  # pragma: no cover


def get_links(model='COCO'):
    if model == 'COCO':
        return [
            (HKP.Value('NECK'), HKP.Value('LEFT_SHOULDER')),
            (HKP.Value('LEFT_SHOULDER'), HKP.Value('LEFT_ELBOW')),
            (HKP.Value('LEFT_ELBOW'), HKP.Value('LEFT_WRIST')),
            (HKP.Value('NECK'), HKP.Value('LEFT_HIP')),
            (HKP.Value('LEFT_HIP'), HKP.Value('LEFT_KNEE')),
            (HKP.Value('LEFT_KNEE'), HKP.Value('LEFT_ANKLE')),
            (HKP.Value('NECK'), HKP.Value('RIGHT_SHOULDER')),
            (HKP.Value('RIGHT_SHOULDER'), HKP.Value('RIGHT_ELBOW')),
            (HKP.Value('RIGHT_ELBOW'), HKP.Value('RIGHT_WRIST')),
            (HKP.Value('NECK'), HKP.Value('RIGHT_HIP')),
            (HKP.Value('RIGHT_HIP'), HKP.Value('RIGHT_KNEE')),
            (HKP.Value('RIGHT_KNEE'), HKP.Value('RIGHT_ANKLE')),
            (HKP.Value('NOSE'), HKP.Value('LEFT_EYE')),
            (HKP.Value('LEFT_EYE'), HKP.Value('LEFT_EAR')),
            (HKP.Value('NOSE'), HKP.Value('RIGHT_EYE')),
            (HKP.Value('RIGHT_EYE'), HKP.Value('RIGHT_EAR')),
        ]
    return []


def get_face_parts(model='COCO'):
    if model == 'COCO':
        return [
            HKP.Value('NOSE'),
            HKP.Value('LEFT_EYE'),
            HKP.Value('LEFT_EAR'),
            HKP.Value('RIGHT_EYE'),
            HKP.Value('RIGHT_EAR'),
        ]
    return []


def get_links_colors():
    return list(permutations([0, 255, 85, 170], 3))


def get_np_image(input_image):
    if isinstance(input_image, np.ndarray):
        return input_image
    if isinstance(input_image, Image):
        buffer = np.frombuffer(input_image.data, dtype=np.uint8)
        return cv2.imdecode(buffer, flags=cv2.IMREAD_COLOR)
    return np.array([], dtype=np.uint8)


def get_pb_image(input_image, encode_format='.jpeg', compression_level=0.8):
    if isinstance(input_image, np.ndarray):
        if encode_format == '.jpeg':
            params = [cv2.IMWRITE_JPEG_QUALITY, int(compression_level * 100)]
        elif encode_format == '.png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, int(compression_level * 9)]
        else:
            return Image()
        ok, buf = cv2.imencode(ext=encode_format, img=input_image, params=params)
        if not ok:
            return Image()
        return Image(data=buf.tobytes())
    if isinstance(input_image, Image):
        return input_image
    return Image()


def draw_skeletons(input_image, skeletons: ObjectAnnotations):
    image = get_np_image(input_image)
    links = get_links()
    face_parts = get_face_parts()
    colors = get_links_colors()
    for ob in skeletons.objects:
        parts = {part.id: (int(part.position.x), int(part.position.y))
                 for part in ob.keypoints}
        for (begin, end), color in zip(links, colors):
            if begin in parts and end in parts:
                cv2.line(image, parts[begin], parts[end], color=color, thickness=4)
        for ptype, center in parts.items():
            radius = 2 if ptype in face_parts else 4
            cv2.circle(image, center=center, radius=radius,
                       color=(255, 255, 255), thickness=-1)
    return image
