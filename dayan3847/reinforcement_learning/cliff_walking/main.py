import numpy as np
from Agent import Environment, AgentQLearning, AgentQLearningTable
from Trainner import Trainner
from flask import Flask, jsonify
from flask_cors import CORS

np.random.seed(0)

if __name__ == '__main__':
    env: Environment = Environment()
    agent: AgentQLearning = AgentQLearningTable(env)
    trainer = Trainner(agent)

    trainer.train()

    app = Flask(__name__)


    @app.route('/api/plot/model', methods=['GET'])
    def get_plot_model():
        global trainer
        return jsonify(trainer.get_status())


    CORS(app, resources={r"/api/*": {"origins": r"*"}})
    app.run(debug=True)
