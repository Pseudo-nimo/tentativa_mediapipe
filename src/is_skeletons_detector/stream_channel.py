from is_wire.core import Channel
from is_wire.core.wire.conversion import WireV1


class StreamChannel(Channel):
    """Channel subclass that always delivers the most recent message.

    When messages arrive faster than they are processed, all buffered
    messages are drained and only the latest one is returned.  The
    number of dropped messages is optionally reported.
    """

    def consume(self, return_dropped=False):
        def _drain(timeout=None):
            self.amqp_message = None
            while self.amqp_message is None:
                self.connection.drain_events(timeout=timeout)
            return self.amqp_message

        amqp_msg = _drain()
        dropped = 0
        while True:
            try:
                amqp_msg = _drain(timeout=0.0)
                dropped += 1
            except Exception:
                msg = WireV1.from_amqp_message(amqp_msg)
                return (msg, dropped) if return_dropped else msg
