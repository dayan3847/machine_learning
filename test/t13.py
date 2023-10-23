import time
import threading
import numpy as np

from flask import Flask, jsonify

app = Flask(__name__)


class Test13:
    _instance = None

    def __init__(self):
        self.count = 0

    @staticmethod
    def get_instance():
        if Test13._instance is None:
            Test13._instance = Test13()
        return Test13._instance

    def run(self):
        while True:
            self.count += 1
            time.sleep(1)


@app.route('/', methods=['GET'])
def hello():
    return jsonify(
        {
            "message": "Hello, World!",
            'count': Test13.get_instance().count,
            'array': np.array([[1, 2, 3], [1, 2, 3]]).tolist(),
        }
    )


if __name__ == '__main__':
    server = threading.Thread(
        target=app.run,
    )
    server.start()

    instance = Test13.get_instance()
    print('Hello, World!')
    instance.run()
