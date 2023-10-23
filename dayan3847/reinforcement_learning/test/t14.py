from flask import Flask, jsonify


class HelloWorldResource:
    def get(self):
        return jsonify({"message": "Hello, World!",
                        "data": [1, 2, 3]})

    def post(self):
        return jsonify({
            "message": "Hello, World!",
            "data": [1, 2, 3]
        })


if __name__ == '__main__':
    app = Flask(__name__)

    h = HelloWorldResource()
    app.add_url_rule('/hello', view_func=h.get, methods=['GET'])
    app.add_url_rule('/hello', view_func=h.post, methods=['POST'])

    app.run(debug=True)
