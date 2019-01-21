"""
@author Jacob
@time 2019/01/16
"""

import typing
import math
import numpy as np


class ErrorFunctionType(object):
    """
    An error function and its derivative.
    """

    @staticmethod
    def error(output: float, target: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def der(output: float, target: float) -> float:
        raise NotImplementedError()


class ActivationFunctionType(object):
    """
    A node's activation function and its derivative.
    """

    @staticmethod
    def output(input: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def der(input: float) -> float:
        raise NotImplementedError()


class RegularizationFunctionType(object):
    """
    Function that computes a penalty cost for a given weight
    in the network.
    """

    @staticmethod
    def output(weight: float) -> float:
        raise NotImplementedError()

    @staticmethod
    def der(weight: float) -> float:
        raise NotImplementedError()


class Errors(object):
    """
    Built-in error functions
    """

    class SQUARE(ErrorFunctionType):
        @staticmethod
        def error(output: float, target: float) -> float:
            return .5 * ((output - target) ** 2)

        @staticmethod
        def der(output: float, target: float) -> float:
            return output - target


class Activations(object):
    """
    Built-in activation functions
    """

    class TANH(ActivationFunctionType):
        @staticmethod
        def output(input: float) -> float:
            return math.tanh(input)

        @staticmethod
        def der(input: float) -> float:
            return 1 - math.tanh(input) ** 2

    class RELU(ActivationFunctionType):
        @staticmethod
        def output(input: float) -> float:
            return max([0, input])

        @staticmethod
        def der(input: float) -> float:
            return 0 if input <= 0 else 1

    class SIGMOID(ActivationFunctionType):
        @staticmethod
        def output(input: float) -> float:
            return 1 / (1 + math.exp(-input))

        @staticmethod
        def der(input: float) -> float:
            r = 1 / (1 + math.exp(-input))
            return r * (1 - r)

    class LINEAR(ActivationFunctionType):
        @staticmethod
        def output(input: float) -> float:
            return input

        @staticmethod
        def der(input: float) -> float:
            return 1


class RegularizationFunction(object):
    """

    """

    class L1(RegularizationFunctionType):
        @staticmethod
        def output(weight: float):
            return abs(weight)

        @staticmethod
        def der(weight: float):
            if weight < 0:
                return -1
            elif weight > 0:
                return 1
            else:
                return 0

    class L2(RegularizationFunctionType):
        @staticmethod
        def output(weight: float):
            return .5 * (weight ** 2)

        @staticmethod
        def der(weight: float):
            return weight


class Node(object):
    """
    A node in a neural network. Each node has a state
    (total input, output, and their respectively derivatives)
    which changes after every forward and back propagation run.
    """

    def __init__(self, id_: str, activation: ActivationFunctionType, init_zero: bool):
        """
        Creates a new node with the provided id and activation function.
        :param id_:
        :param activation:
        :param init_zero:
        """

        self.id = id_
        """
        Activation function that takes total input and returns node's
        output.
        """
        self.activation = activation
        self.bias = 0 if init_zero else .1

        """
        list of input links
        """
        self.input_links: typing.List[Link] = []
        self.total_input: float = None
        """
        list of output links
        """
        self.outputs: typing.List[Link] = []
        self.output: float = None
        """
        error derivative with respect to this node's output
        """
        self.output_der = 0
        """
        error derivative with respect to this node's total input
        """
        self.input_der = 0
        """
        Accumulated error derivative with respect to this node's
        total input since the last update. This derivative equals
        dE/db where b is the node's bias term.
        """
        self.acc_input_der = 0
        """
        Number of accumulated err. derivatives with respect to the
        total input since the last update.
        """
        self.num_accumulated_ders = 0

    def update_output(self):
        """
        Recomputes the node's output and returns it.
        :return:
        """
        self.total_input = self.bias
        for i in range(len(self.input_links)):
            link = self.input_links[i]
            self.total_input += link.weight * link.source.output
        self.output = self.activation.output(self.total_input)
        return self.output


class Link(object):
    """
    A link in a neural network. Each link has a weight and a source and
    destination node. Also it has an internal state (error derivative
    with respect to a particular input) which gets updated after a
    run of back propagation.
    """

    def __init__(self, source: Node, dest: Node, regularization: RegularizationFunctionType, init_zero: bool):
        """
        Constructs a link in the neural network initialized with random weight

        :param source: The source node.
        :param dest: The destination node.
        :param regularization: The regularization function that computes
            the penalty for this weight. If null, there will be no regularization.
        :param init_zero:
        """

        self.dest = dest
        self.regularization = regularization
        self.weight = 0 if init_zero else np.random.random() - .5

        self.source = source
        self.id = source.id
        self.is_dead = False
        """
        Error derivative with respect to this weight.
        """
        self.error_der = 0
        """
        Accumulated error derivative since the last update.
        """
        self.acc_error_der = 0
        """
        Number of accumulated derivatives since the last update.
        """
        self.num_accumulated_ders = 0


node_layer = typing.List[Node]
node_matrix = typing.List[node_layer]


def build_network(network_shape: typing.List[float],
                  activation: ActivationFunctionType,
                  output_activation: ActivationFunctionType,
                  regularization: RegularizationFunctionType,
                  input_ids: typing.List[str],
                  init_zero: bool) -> node_matrix:
    """
    Builds a neural network.

    :param network_shape: The shape of the network. E.g. [1, 2, 3, 1] means
        the network will have one input node, 2 nodes in first hidden layer,
        3 nodes in second hidden layer and 1 output node.
    :param activation: The activation function of every hidden node.
    :param output_activation: The activation function for the output node.
    :param regularization: The regularization function that computes a penalty
        for a given weight (parameter) in the network. If null, there will be
        no regularization.
    :param input_ids: List of ids for the input nodes.
    :param init_zero:
    :return:
    """

    num_layers = len(network_shape)
    id_ = 1
    network: node_matrix = []
    for layer_idx in range(num_layers):
        is_output_layer = layer_idx == num_layers - 1
        is_input_layer = layer_idx == 0
        current_layer: node_layer = []
        network.append(current_layer)
        """
        List of layers, with each layer being a list of nodes.
        """
        num_nodes = network_shape[layer_idx]
        for i in range(int(num_nodes)):
            node_id = str(id_)
            if is_input_layer:
                node_id = input_ids[i]
            else:
                id_ += 1
            node = Node(node_id, output_activation if is_output_layer else activation, init_zero)
            current_layer.append(node)
            if layer_idx >= 1:
                for j in range(len(network[layer_idx - 1])):
                    prev_node = network[layer_idx - 1][j]
                    link = Link(prev_node, node, regularization, init_zero)
                    prev_node.outputs.append(link)
                    node.input_links.append(link)

    return network


def forward_prop(network: node_matrix, inputs: typing.List[float]) -> float:
    """
    Runs a forward propagation of the provided input through the provided
    network. This method modifies the internal state of the net network - the
    total input and output of each node in the network.

    :param network: The neural network.
    :param inputs: The input array. Its length should match the number of
        input nodes in the network.
    :return: The final output of the network.
    """
    input_layer = network[0]
    if len(inputs) != len(input_layer):
        raise Exception("The number of inputs must match the number of nodes in the input layer")

    """
    Update the input layer.
    """
    for i in range(len(input_layer)):
        node = input_layer[i]
        node.output = inputs[i]
    for layer_idx in range(1, len(network)):
        current_layer = network[layer_idx]
        """
        Update all the nodes in this layer
        """
        for i in range(len(current_layer)):
            node = current_layer[i]
            node.update_output()

    return network[len(network) - 1][0].output


def back_prop(network: node_matrix, target: float, error_func: ErrorFunctionType):
    """
    Runs a backward propagation using the provided target and the
    computed output of the previous call to forward propagation.
    This method modifies the internal state of the network - the
    error derivatives with respect to each node, and each weight
    in the network.

    :param network:
    :param target:
    :param error_func:
    :return:
    """

    """
    The output node is a special case. We use the user-defined error
    function for the derivative.
    """
    output_node = network[len(network) - 1][0]
    output_node.output_der = error_func.der(output_node.output, target)
    """
    Go through the layers backwards.
    """
    for layer_idx in range(len(network) - 1, 0, -1):
        current_layer = network[layer_idx]
        """
        Compute the error derivative of each node with respect to:
        1) its total input
        2) each of its input weights
        """
        for i in range(len(current_layer)):
            node = current_layer[i]
            node.input_der = node.output_der * node.activation.der(node.total_input)
            node.acc_input_der += node.input_der
            node.num_accumulated_ders += 1

        """
        Error derivative with respect to each weight coming into the node.
        """
        for i in range(len(current_layer)):
            node = current_layer[i]
            for j in range(len(node.input_links)):
                link = node.input_links[i]
                if link.is_dead:
                    continue
                link.error_der = node.input_der * link.source.output
                link.acc_error_der += link.error_der
                link.num_accumulated_ders += 1
        if layer_idx == 1:
            continue
        prev_layer = network[layer_idx - 1]
        for i in range(len(prev_layer)):
            node = prev_layer[i]
            """
            Compute the error derivative with respect to each node's output
            """
            for j in range(len(node.outputs)):
                output = node.outputs[j]
                node.output_der += output.weight * output.dest.input_der


def update_weights(network: node_matrix, learning_rate: float, regularization_rate: float):
    """
    Updates the weights of the network using the previously accumulated error derivatives.

    :param network:
    :param learning_rate:
    :param regularization_rate:
    :return:
    """

    for layer_idx in range(1, len(network)):
        current_layer = network[layer_idx]
        for i in range(len(current_layer)):
            node = current_layer[i]
            """
            Update the node's bias.
            """
            if node.num_accumulated_ders > 0:
                node.bias -= learning_rate * node.acc_input_der / node.num_accumulated_ders
                node.acc_input_der = 0
                node.num_accumulated_ders = 0
            """
            Update the weights coming into this node.
            """
            for j in range(len(node.input_links)):
                link = node.input_links[j]
                if link.is_dead:
                    continue
                regul_der = link.regularization.der(link.weight) if link.regularization else 0
                if link.num_accumulated_ders > 0:
                    """
                    Update the weight based on dE/dw
                    """
                    link.weight = link.weight - (learning_rate / link.num_accumulated_ders) * link.acc_error_der
                    new_link_weight = link.weight - (learning_rate * regularization_rate) * regul_der
                    if link.regularization == RegularizationFunction.L1 and link.weight * new_link_weight < 0:
                        link.weight = 0
                        link.is_dead = True
                    else:
                        link.weight = new_link_weight
                    link.acc_error_der = 0
                    link.num_accumulated_ders = 0


def for_each_node(network: node_matrix, ignore_inputs: bool, accessor):
    """
    Iterates over every node in the network.

    :param network:
    :param ignore_inputs:
    :param accessor:
    :return:
    """
    if ignore_inputs:
        start = 1
    else:
        start = 0
    for layer_idx in range(start, len(network)):
        current_layer = network[layer_idx]
        for i in range(len(current_layer)):
            node = current_layer[i]
            accessor(node)


def get_output_node(network: node_matrix):
    """
    Returns the output node in the network.

    :param network:
    :return:
    """
    return network[len(network) - 1][0]


