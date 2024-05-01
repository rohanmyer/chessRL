from keras.models import Sequential
from keras.layers import (
    Input,
    Dense,
    Flatten,
    Concatenate,
    Conv2D,
    Dropout,
    Activation,
    BatchNormalization,
    LeakyReLU,
)
from keras.layers import add as add_layer
from keras.losses import mean_squared_error
from keras.models import Model, clone_model, load_model
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
import chess.engine


class EngineAgent(object):
    def __init__(self, engine_path, color="black"):
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.limit = chess.engine.Limit(depth=10)
        self.color = color

    def predict(self, board):
        result = self.engine.play(board, self.limit)
        return result.move


class Agent(object):

    def __init__(self, lr=0.01, network="big"):
        self.optimizer = Adam(learning_rate=lr)
        self.model = Model()
        self.proportional_error = False
        self.network = network
        if network == "big":
            self.init_bignet()
        else:
            self.init_network()

    def fix_model(self):
        """
        The fixed model is the model used for bootstrapping
        Returns:
        """

        self.fixed_model = clone_model(self.model)
        self.fixed_model.compile(optimizer=self.optimizer, loss="mse", metrics=["mae"])
        self.fixed_model.set_weights(self.model.get_weights())

    def init_network(self):
        """
        Builds the neural network architecture
        """
        main_input = Input(shape=(8, 8, 8), name="main_input")

        x = self.build_convolutional_layer(main_input)

        # add a high amount of residual layers
        for i in range(20):
            x = self.build_residual_layer(x)

        x = self.build_value_head(x)

        self.model = Model(inputs=main_input, outputs=x, name="chess_model")

        self.model.compile(optimizer=self.optimizer, loss=mean_squared_error)

    def build_convolutional_layer(self, input_layer):
        """
        Builds a convolutional layer
        """

        layer = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
            use_bias=False,
        )(input_layer)
        layer = BatchNormalization(axis=1)(layer)
        layer = Activation("relu")(layer)
        return layer

    def build_residual_layer(self, input_layer):
        """
        Builds a residual layer
        """
        # first convolutional layer
        layer = self.build_convolutional_layer(input_layer)
        # second convolutional layer with skip connection
        layer = Conv2D(
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
            use_bias=False,
        )(layer)
        layer = BatchNormalization(axis=1)(layer)
        # skip connection
        layer = add_layer([layer, input_layer])
        # activation function
        layer = Activation("relu")(layer)
        return layer

    def build_value_head(self, input) -> Model:
        """
        Builds the value head of the neural network
        """
        layer = Conv2D(
            1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            data_format="channels_first",
        )(input)
        layer = BatchNormalization(axis=1)(layer)
        layer = Activation("relu")(layer)
        layer = Flatten()(layer)
        layer = Dense(256)(layer)
        layer = Activation("relu")(layer)
        # output shape == 1, because we want 1 value: the estimated outcome from the position
        # tanh activation function maps the output to [-1, 1]
        layer = Dense(1, activation="tanh")(layer)
        return layer

    def init_bignet(self):
        layer_state = Input(shape=(8, 8, 8), name="state")
        conv_xs = Conv2D(4, (1, 1), activation="relu")(layer_state)
        conv_s = Conv2D(8, (2, 2), strides=(1, 1), activation="relu")(layer_state)
        conv_m = Conv2D(12, (3, 3), strides=(2, 2), activation="relu")(layer_state)
        conv_l = Conv2D(16, (4, 4), strides=(2, 2), activation="relu")(layer_state)
        conv_xl = Conv2D(20, (8, 8), activation="relu")(layer_state)
        conv_rank = Conv2D(3, (1, 8), activation="relu")(layer_state)
        conv_file = Conv2D(3, (8, 1), activation="relu")(layer_state)

        f_xs = Flatten()(conv_xs)
        f_s = Flatten()(conv_s)
        f_m = Flatten()(conv_m)
        f_l = Flatten()(conv_l)
        f_xl = Flatten()(conv_xl)
        f_r = Flatten()(conv_rank)
        f_f = Flatten()(conv_file)

        dense1 = Concatenate(name="dense_bass")([f_xs, f_s, f_m, f_l, f_xl, f_r, f_f])
        dense2 = Dense(256, activation="sigmoid")(dense1)
        dense3 = Dense(128, activation="sigmoid")(dense2)
        dense4 = Dense(56, activation="sigmoid")(dense3)
        dense5 = Dense(64, activation="sigmoid")(dense4)
        dense6 = Dense(32, activation="sigmoid")(dense5)

        value_head = Dense(1)(dense6)

        self.model = Model(inputs=layer_state, outputs=value_head)
        self.model.compile(optimizer=self.optimizer, loss=mean_squared_error)

    def predict_distribution(self, states, batch_size=256):
        """
        :param states: list of distinct states
        :param n:  each state is predicted n times
        :return:
        """
        predictions_per_state = int(batch_size / len(states))
        state_batch = []
        for state in states:
            state_batch = state_batch + [state for x in range(predictions_per_state)]

        state_batch = np.stack(state_batch, axis=0)
        predictions = self.model.predict(state_batch, verbose=0)
        predictions = predictions.reshape(len(states), predictions_per_state)
        mean_pred = np.mean(predictions, axis=1)
        std_pred = np.std(predictions, axis=1)
        upper_bound = mean_pred + 2 * std_pred

        return mean_pred, std_pred, upper_bound

    def predict(self, board_layer):
        return self.model.predict(board_layer, verbose=0)

    def predict_batch(self, board_layers):
        return self.model.predict_on_batch(board_layers)

    def TD_update(self, states, rewards, sucstates, episode_active, gamma=0.9):
        """
        Update the SARSA-network using samples from the minibatch
        Args:
            minibatch: list
                The minibatch contains the states, moves, rewards and new states.

        Returns:
            td_errors: np.array
                array of temporal difference errors

        """
        suc_state_values = self.fixed_model.predict(sucstates, verbose=0)
        V_target = np.array(rewards) + np.array(episode_active) * gamma * np.squeeze(
            suc_state_values
        )
        # Perform a step of minibatch Gradient Descent.
        self.model.fit(x=states, y=V_target, epochs=1, verbose=0)

        V_state = self.model.predict(states, verbose=0)  # the expected future returns
        td_errors = V_target - np.squeeze(V_state)

        return td_errors

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        print(f"Loading model from {path}")
        self.model = load_model(path)
        self.model.compile(optimizer=self.optimizer, loss=mean_squared_error)
        self.fix_model()
