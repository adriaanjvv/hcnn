import tensorflow as tf
import numpy as np
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from keras.layers.recurrent import SimpleRNN
from keras.optimizers

from rnn_cell import BasicHCNNCell


def hcnn_loss(y, y_predicted):
    return tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(y, y_predicted)), axis=1), axis=1)

class HcnnModel(object):
    """
    Define an interface to instantiate, train and forecast with a model.
    What should forecast be?
    - The ensemble output, or
    - some sensible averaging?
    -- Feel most usefull/flexible output is ensemble output.
    -For now go with ensemble output.
    What should input be?
    - Ensemble replication of input, or
    - just single input?
    -- Still makes sense to output ensemble but input only one replicate of input.
    -For now go with Single input, thus replication is internal to model (internal to layer).
    """
    def __init__(selfs, input, output):
        # Input is tensor of numpy array - could consider making input layer that takes numpy to tensor, or use keras layer, think
        # that is what keras input layer does.
        self.input = input
        self.output = output


    def compile(self, optimizer, loss_function, metrics=None):
        # Take output from layer, send to loss function, send to optimizer, create predictions, send predictions result to calc metrics
        # Should call build functions of layers.
        input_shape = tf.shape(self.input)
        loss = loss_function(self.input,tf.slice(self.output, begin=(0,0,0), size=(input_shape[0],input_shape[1],input_shape[2])))
        global_step = tf.Variable(0, trainable=False)
        self.train_step = optimizer.minimize(loss, global_step)

        init = tf.initialize_all_variables()
        tf.get_default_graph().finalize() #throws error if nodes are added to graph after this point
        self.sess = tf.Session()
        return self.sess.run(init)

    def fit(self, x=None, y=None, epochs=None):
        # train a number of epochs on x.
        # y is x + the forecast.

        with tf.variable_scope('hcnn'):
            self.layer.

        pass

    def evaluate(self, x=None, y=None):
        # return the loss and metrics.
        # x is the input. Y is x + the forecast values.
        pass

    def predict(self, x=None):
        # return y, which is x + the forecast values

        pass


class HcnnLayer(object):
    """
    Define a layer that can be fed into HcnnModel.
    """
    def __init__(self, output_dim,  state_size , density=0.3 , ensemble_size=1, **kwargs):
        # Return self.
        self.output_dim = output_dim
        self.num_units = state_size
        self.ensemble_size = ensemble_size
        self.density = density
        self.cell = None
        self.sparse_cube = None
        self.input_shape = None
        return self

    def _signal_shape(self, input_shape):
        # return signal_length and signal dimension, e.g number of time points and number of securities
        return {'signal_length': input_shape[0], 'signal_dimension': input_shape[1]}

    def build(self, input_shape):
        # Create sparse weight matrix for ensemble
        # Create a trainable weight variable for this layer.
        # Create the tensorflow graph - unfold rnn cells.
        # Session create and init should happen here

        self.input_shape = self._signal_shape(input_shape)


        self.cell = BasicHCNNCell(num_units=self.num_units , ensemble_size=self.ensemble_size, density=self.density)
        # TODO: sparse_cube should only be needed on grads for this layer, yet I am accessing it outside
        self.sparse_cube = self.cell.sparse_cube
        return self.cell


    def call(self, input, noise):
        # input tensor
        # noise - tensor
        cell = self.build(tf.shape(input))
        self.initial_state = tf.Variable(np.zeros(tuple(self.cell.state_shape),dtype='float32'))
        self.state = self.initial_state+noise

        output_shape = self.compute_output_shape(input_shape)
        output_list = []
        forecast_list = []

        with tf.variable_scope('hcnn'):
            for i in range(self.input_shape.signal_length):
                if i == 1:
                    vs.get_variable_scope().reuse_variables()

                self.state, output = self.cell(self.state, tf.slice(input ,begin=(0,i,0), size=(self.ensemble_size, 1,output_shape[2])))
                output_list.append(output)

            final_state = self.state

            ### Extend to forecast ###
            forecast_state = final_state
            for j in range(output_shape[2] - self.input_shape.signal_length):
                forecast_state, forecast_output = cell(forecast_state)
                forecast_list.append(forecast_output)

            outputs = tf.squeeze(tf.stack(output_list,axis=1),axis=2)
            forecast = tf.concat([outputs, tf.squeeze(tf.stack(forecast_list, axis=1),axis=2)],axis=1)
        return forecast

    def __call__(self, x):
        # Provide HcnnLayer(input) functionality.
        return self.call(x)

    def compute_output_shape(self, input_shape):

        return (self.ensemble_size, self.output_dim, self._signal_shape(input_shape).signal_dimension)

class MaskingOptimizer(GradientDescentOptimizer):
    """
    Class derived from Tensorflow optimizer with sparse masking added.
    """
    def __init__(self, sparse_cube, learning_rate, use_locking=False, name="GradientDescent", clipping_value=1):
        self.clipping_value =  clipping_value
        self.sparse_cube = sparse_cube
        super(masking_optimizer, self).__init__(learning_rate, use_locking, name)

    def minimize(self, loss, global_step=None, var_list=None,
               gate_gradients=GATE_OP, aggregation_method=None,
               colocate_gradients_with_ops=False, name=None,
               grad_loss=None):
        """
        Add operations to minimize loss by updating var_list.
        :param loss: A `Tensor` containing the variable to minimize.
        :param global_step: Optional `Variable` to increment by one after the
        variables have been updated.
        :param var_list: Optional list of `Variable` objects to update to minimize
        `loss`.  Defaults to the list of variables collected in the graph
        under the key `GraphKeys.TRAINABLE_VARIABLES`.
        :param gate_gradients:
        :param aggregation_method:
        :param colocate_gradients_with_ops:
        :param name:
        :param grad_loss:
        :return:
        """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)


        # Apply mask and clipping to gradients
        new_list=[]
        for grad, var in grads_and_vars:

            if 'weights' in var.name:
                new_list.append((tf.clip_by_value(grad, -self.clipping_value, self.clipping_value)*self.sparse_cube,var))
            else:
                new_list.append((grad, var))
        grads_and_vars_masked = new_list

        return self.apply_gradients(grads_and_vars_masked, global_step = global_step)
