from .stream_wrapper import StreamWrapper
from .debug import Debug
import socket
import sys


class Agent:
    def __init__(self, host="127.0.0.1", port=31000, token="0"*16):
        self.socket = socket.socket()
        self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        while True:
            try:
                self.socket.connect((host, port))
                break
            except Exception:
                continue
        socket_stream = self.socket.makefile('rwb')
        self.reader = StreamWrapper(socket_stream)
        self.writer = StreamWrapper(socket_stream)
        self.token = token
        self.writer.write_string(self.token)
        self.writer.flush()
