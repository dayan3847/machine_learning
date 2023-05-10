from multi_layer_perceptron.src.net.Neuron import Neuron


class NeuronInput(Neuron):

    def get_value(self) -> float:
        if self._value is None:
            raise Exception("Input node has no value")
        return self._value

    def set_value(self, value: float):
        self._value = value
