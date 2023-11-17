import numpy as np

from dayan3847.models.Model import Model
from dayan3847.models.multivariate.MultivariateGaussianModel import ModelGaussianMultivariate
from dayan3847.models.ModelTrainer import ModelTrainer

if __name__ == '__main__':
    # the data would be in a 3xN matrix
    # where 3 is the dimension of the data and N is the number of data
    # de los 3, el ultimo es de Y y los anteriores forman el vector X
    data: np.array = np.loadtxt('../../2_1_gaussian_sigmoidal_basis_functions/data_3d.csv',
                                delimiter=',').T  # Load Data
    model: Model = ModelGaussianMultivariate(.1, (7, 7), ((0, 1), (0, 1)), .01)
    trainer: ModelTrainer = ModelTrainer(model, data, 100)

    app = Flask(__name__)
    CORS(app)


    @app.route('/api/on', methods=['GET'])
    def on():
        global trainer
        if not trainer.running:
            trainer.running = True
            trainer.train()
        return jsonify({'message': 'on'})


    @app.route('/api/off', methods=['GET'])
    def off():
        global trainer
        trainer.running = False
        return jsonify({'message': 'off'})


    @app.route('/api/plot/dataset', methods=['GET'])
    def get_plot_dataset():
        global trainer

        return jsonify({
            'dataset': {
                'data': [
                    {
                        'x': trainer.data_x[0].tolist(),
                        'y': trainer.data_x[1].tolist(),
                        'z': trainer.data_y.tolist(),
                        'mode': 'markers',
                        'type': 'scatter3d',
                        'marker': {
                            'opacity': 0.8,
                        },
                    },
                ]
            },
        })


    @app.route('/api/plot/model', methods=['GET'])
    def get_plot_model():
        global trainer

        _x, _y, _z = trainer.model.data_to_plot_plotly()

        return jsonify({
            'epoch': trainer.current_epoch,
            'error': {
                'data': [
                    {
                        'y': trainer.error_history.tolist(),
                        'mode': 'lines',
                        'marker': {'color': 'red'},
                    }
                ]
            },
            'model': {
                'data': [
                    {
                        'x': trainer.data_x[0].tolist(),
                        'y': trainer.data_x[1].tolist(),
                        'z': trainer.data_y.tolist(),
                        'mode': 'markers',
                        'type': 'scatter3d',
                        'marker': {
                            'opacity': .3,
                        },
                    },
                    {
                        'x': _x.tolist(),
                        'y': _y.tolist(),
                        'z': _z.tolist(),
                        'type': 'surface',
                    },
                ]
            }
        })


    CORS(app, resources={r"/api/*": {"origins": r"*"}})
    app.run(debug=True)
