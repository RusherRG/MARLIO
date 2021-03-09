from .model import PlayerMessageGame


class Debug:
    def __init__(self, writer):
        self.writer = writer

    def draw(self, data):
        PlayerMessageGame.CustomDataMessage(data).write_to(self.writer)
        self.writer.flush()
