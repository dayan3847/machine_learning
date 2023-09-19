import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import plotly.graph_objects as go
import threading
import dash

iris = datasets.load_iris()
X = iris.data
target = iris.target
Y = keras.utils.to_categorical(target, dtype="uint8")

idx = 0
print(X[idx])
print(target[idx])
print(Y[idx])

neuron_per_layer: int = 3
hidden_layers_count: int = 1
activation_fun: str = "sigmoid"

model_layers = \
    [keras.layers.Dense(3, activation="sigmoid", name="layer_in", input_shape=(4,))] \
    + [
        keras.layers.Dense(neuron_per_layer, activation=activation_fun, name=f"layer_hide_{i}")
        for i in range(hidden_layers_count)
    ] \
    + [keras.layers.Dense(3, activation="softmax", name="layer_out")]

len(model_layers)

model = keras.Sequential(model_layers)

model.summary()

y = model(X)
print(y[149])


class Data:
    e = np.array([])
    l0n0wb = np.array([])
    l0n0w0 = np.array([])
    l0n0w1 = np.array([])
    l0n0w2 = np.array([])
    l0n0w3 = np.array([])
    l0n0w4 = np.array([])

    l1n0wb = np.array([])
    l1n0w0 = np.array([])
    l1n0w1 = np.array([])
    l1n0w2 = np.array([])
    l1n0w3 = np.array([])


class WCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        Data.e = np.append(Data.e, epoch)
        weights = self.model.get_weights()
        Data.l0n0wb = np.append(Data.l0n0wb, weights[1][0])
        Data.l0n0w0 = np.append(Data.l0n0w0, weights[0][0][0])
        Data.l0n0w1 = np.append(Data.l0n0w1, weights[0][1][0])
        Data.l0n0w2 = np.append(Data.l0n0w2, weights[0][2][0])
        Data.l0n0w3 = np.append(Data.l0n0w3, weights[0][3][0])

        Data.l1n0wb = np.append(Data.l1n0wb, weights[3][0])
        Data.l1n0w0 = np.append(Data.l1n0w0, weights[2][0][0])
        Data.l1n0w1 = np.append(Data.l1n0w1, weights[2][1][0])
        Data.l1n0w2 = np.append(Data.l1n0w2, weights[2][2][0])


model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.03),
    loss=keras.losses.MeanSquaredError()
)


def get_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l0n0wb, name="l0n0wb"))
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l0n0w0, name="l0n0w0"))
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l0n0w1, name="l0n0w1"))
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l0n0w2, name="l0n0w2"))
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l0n0w3, name="l0n0w3"))

    fig.add_trace(go.Scatter(x=Data.e, y=Data.l1n0wb, name="l1n0wb"))
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l1n0w0, name="l1n0w0"))
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l1n0w1, name="l1n0w1"))
    fig.add_trace(go.Scatter(x=Data.e, y=Data.l1n0w2, name="l1n0w2"))


    # axis names
    fig.update_xaxes(title_text="epoch")
    fig.update_yaxes(title_text="weight")
    return fig


app = dash.Dash(__name__)
app.layout = dash.html.Div([
    dash.dcc.Graph(id='scatter-plot', figure=get_figure(),
                   style={'width': '100%', 'display': 'inline-block', 'height': 1000, }),
    dash.dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
], style={'width': '100%', 'display': 'inline-block', 'height': 1000, })


@app.callback(
    dash.dependencies.Output('scatter-plot', 'figure'),
    dash.dependencies.Input('interval-component', 'n_intervals'),
)
def update_scatter_plot(n):
    return get_figure()


def run_server():
    app.run_server(debug=True)


def fix():
    w_callback = WCallback()
    return model.fit(X, Y, epochs=10000, callbacks=[w_callback])


t_server = threading.Thread(target=run_server, args=())
t_fix = threading.Thread(target=fix, args=())

# t_server.start()
t_fix.start()

run_server()
