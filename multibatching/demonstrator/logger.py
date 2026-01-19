import sys
import os

# Path Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)


class StreamlitLogger:
    def __init__(self, container):
        self.container = container
        self.buffer = []
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)
        self.buffer.append(message)
        self.container.code("".join(self.buffer), language="text")

    def flush(self):
        self.terminal.flush()
        
    def get_content(self):
        return "".join(self.buffer)