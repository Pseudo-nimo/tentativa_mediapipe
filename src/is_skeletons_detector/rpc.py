from functools import partial

from is_wire.core import Channel, Status, StatusCode, Logger
from is_wire.core import Tracer, AsyncTransport
from opencensus.ext.zipkin.trace_exporter import ZipkinExporter
from is_wire.rpc import ServiceProvider, LogInterceptor, TracingInterceptor
from is_msgs.image_pb2 import Image, ObjectAnnotations

from .skeletons import SkeletonsDetector
from .utils import load_options


class _RPCHandler:
    def __init__(self, detector: SkeletonsDetector):
        self._sd = detector

    def detect(self, image: Image, ctx) -> ObjectAnnotations:
        try:
            return self._sd.detect(image)
        except Exception:
            return Status(code=StatusCode.INTERNAL_ERROR)


def main():
    service_name = 'SkeletonsDetector.Detect'

    op = load_options()
    handler = _RPCHandler(SkeletonsDetector(op))

    log = Logger(name=service_name)
    channel = Channel(op.broker_uri)
    log.info('Connected to broker {}', op.broker_uri)

    provider = ServiceProvider(channel)
    provider.add_interceptor(LogInterceptor())

    max_batch_size = max(100, op.zipkin_batch_size)
    exporter = ZipkinExporter(
        service_name=service_name,
        host_name=op.zipkin_host,
        port=op.zipkin_port,
        transport=AsyncTransport,
    )
    provider.add_interceptor(TracingInterceptor(exporter=exporter))

    provider.delegate(
        topic='SkeletonsDetector.Detect',
        function=partial(_RPCHandler.detect, handler),
        request_type=Image,
        reply_type=ObjectAnnotations,
    )

    provider.run()


if __name__ == '__main__':
    main()
