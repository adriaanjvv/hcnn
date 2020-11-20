import pandas as pd
import numpy as np
from scipy.sparse import rand as rand_sparse_matrix
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh
from tensorflow.nn import relu
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.init_ops import UniformUnitScaling
from tensorflow.python.framework import dtypes


class SparseUniformUnitInitializer(UniformUnitScaling):

    def __init__(self, sparse_cube, factor=1.0, seed=None, dtype=dtypes.float32):
        self.sparse_cube = sparse_cube
        super(SparseUniformUnitInitializer, self).__init__(factor=factor, seed=seed, dtype=dtype)

    def __call__(self, shape, dtype=None, partition_info=None):
        var = super(SparseUniformUnitInitializer, self).__call__(shape, dtype, partition_info)

        return var*self.sparse_cube


class RNNCell(object):
  """Abstract object representing an RNN cell.

  The definition of cell in this package differs from the definition used in the
  literature. In the literature, cell refers to an object with a single scalar
  output. The definition in this package refers to a horizontal array of such
  units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  tuple of integers, then it results in a tuple of `len(state_size)` state
  matrices, each with a column size corresponding to values in `state_size`.

  This module provides a number of basic commonly used RNN cells, such as
  LSTM (Long Short Term Memory) or GRU (Gated Recurrent Unit), and a number
  of operators that allow add dropouts, projections, or embeddings for inputs.
  Constructing multi-layer cells is supported by the class `MultiRNNCell`,
  or by calling the `rnn` ops several times. Every `RNNCell` must have the
  properties below and and implement `__call__` with the following signature.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.

    Args:
      inputs: `2-D` tensor with shape `[batch_size x input_size]`.
      state: if `self.state_size` is an integer, this should be a `2-D Tensor`
        with shape `[batch_size x self.state_size]`.  Otherwise, if
        `self.state_size` is a tuple of integers, this should be a tuple
        with shapes `[batch_size x s] for s in self.state_size`.
      scope: VariableScope for the created subgraph; defaults to class name.

    Returns:
      A pair containing:
      - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
      - New state: Either a single `2-D` tensor, or a tuple of tensors matching
        the arity and shapes of `state`.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = self.state_size
    if nest.is_sequence(state_size):
      state_size_flat = nest.flatten(state_size)
      zeros_flat = [
          array_ops.zeros(
              array_ops.pack(_state_size_with_prefix(s, prefix=[batch_size])),
              dtype=dtype)
          for s in state_size_flat]
      for s, z in zip(state_size_flat, zeros_flat):
        z.set_shape(_state_size_with_prefix(s, prefix=[None]))
      zeros = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=zeros_flat)
    else:
      zeros_size = _state_size_with_prefix(state_size, prefix=[batch_size])
      zeros = array_ops.zeros(array_ops.pack(zeros_size), dtype=dtype)
      zeros.set_shape(_state_size_with_prefix(state_size, prefix=[None]))

    return zeros


class BasicHCNNCell(RNNCell):
    """
    The most basic Historically Consistent Neural Network (HCNN) cell.
    The inputs on the call method are treated as observations.
    """

    def __init__(self, num_units, ensemble_size=1, output_size=None, input_size=None, density=1):

        if input_size is not None:
            print('Input_size parameter depreciated.')
            # logging.warn("%s: The input_size parameter is deprecated." % self)

        self._num_units = num_units

        self._output_size = None

        if output_size is not None:
            self._output_size = output_size

        self._state_shape = [ensemble_size, 1, num_units]
        self._sparse_cube = _get_sparse(self._state_shape, num_units, density)

        self._state_init_var = UniformUnitScaling()

    def generate_random_state(self):
        return  self._state_init_var(self._state_shape)

    @property
    def state_size(self):
        return self._num_units

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def sparse_cube(self):
        return self._sparse_cube

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, state, inputs=None, scope=None):  # inputs are observations
        """
        Basic HCNN with Architectural Teacher-Forcing (ATF): new_state = A * tanh( state - diff([id 0]*state,expand(inputs)) ).
        : expand(.) zero pads vector to same size as state

        state: dim [depth,1,n]
        inputs: dim [depth,1,m]
        output: dim [depth,1,m]
        """

        # TODO: What is this? Write comment or remove.
        if inputs is not None:
            in_shape = inputs.get_shape()
            in_len = len(in_shape)
            if in_len == 3:
                obs = inputs
            if in_len <= 2:
                obs = tf.expand_dims(inputs, 0)
            if in_len == 1:
                obs = tf.expand_dims(obs, 0)
            if self._output_size is None:
                self._output_size = in_shape[2].value
            elif self._output_size != in_shape[2].value:
                raise ValueError("output_size and input shape are inconsistent.")


        if (self._output_size is None):
            raise ValueError("Output_size is ill defined.")

        if len(state.get_shape()) < 3:
            raise ValueError("State should have 3 dimensions.")

        with vs.variable_scope(scope or type(self).__name__) as cell_scope:  # "BasicHCNNCell"
            # returns first output_size num elems from state as tensor
            output = state[:, :, :self._output_size]

            if inputs is not None:
                # zero pads the difference between output and obs to be of the same size as state
                padding = [[0, 0], [0, 0], [0, self._num_units - self._output_size]]
                tar = tf.pad((tf.subtract(output, obs)), padding)
                # get new state.
                new_state = _linear_cube(relu(state - tar), self._num_units, sparse_cube=self._sparse_cube, scope=cell_scope)
            else:
                # When there is no teacher forcing, as with forecast.
                new_state = _linear_cube(relu(state), self._num_units, sparse_cube=self._sparse_cube, scope=cell_scope)
        return new_state, output


def _state_size_with_prefix(state_size, prefix=None):
  """Helper function that enables int or TensorShape shape specification.

  This function takes a size specification, which can be an integer or a
  TensorShape, and converts it into a list of integers. One may specify any
  additional dimensions that precede the final state size specification.

  Args:
    state_size: TensorShape or int that specifies the size of a tensor.
    prefix: optional additional list of dimensions to prepend.

  Returns:
    result_state_size: list of dimensions the resulting tensor size.
  """
  result_state_size = tensor_shape.as_shape(state_size).as_list()
  if prefix is not None:
    if not isinstance(prefix, list):
      raise TypeError("prefix of _state_size_with_prefix should be a list.")
    result_state_size = prefix + result_state_size
  return result_state_size

def _linear_own(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable(
        "Matrix", [total_arg_size, output_size],
        dtype=dtype,
        initializer=tf.truncated_normal_initializer(0,0.1),  # what is the default behaviour?
        name='inner_rnn_weights'
        )
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=init_ops.constant_initializer(
            bias_start, dtype=dtype))
  return res + bias_term


def _get_sparse(shape, output_size, density):
    """
    Return sparse matrix that is repeated shape[0] times in the first dimension to form a cube.
    Args:
        shape: Shape of input to be manipulated.
        output_size: Length of output.
        density: The density of the sparse matrix.

    Returns: Sparse numpy cube of dimension (shape[0],shape[1],output_size)

    """
    SM = rand_sparse_matrix(shape[2], output_size, density=density)
    sparse_mask = np.array(SM.todense() > 0)
    broadcast_var = np.ones(shape[0])
    cube = sparse_mask[None,:,:]*broadcast_var[:,None,None]
    return cube

def _linear_cube(arg, output_size, sparse_cube, scope):
    """Linear map: arg * W, where W is 3D variable.

      Args:
        args: a 3D Tensor of size depth x 1 x n.
        output_size: int, second dimension of W.
        scope: (optional) Variable scope to create parameters in.

      Returns:
        A 3D Tensor with shape [depth, 1 x output_size] equal to
        arg * W, where W is a newly created matrix.

      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
      """
    if arg is None:
        raise ValueError("`arg` must be specified")

    shape = arg.get_shape()
    arg_size = shape[2].value
    depth = shape[0].value
    dtype = arg.dtype

    # Now the computation.
    with vs.variable_scope(scope):
        weights = vs.get_variable(
            "weights", [depth, arg_size, output_size],
            dtype=dtype,
            initializer = SparseUniformUnitInitializer(sparse_cube=sparse_cube)
        )
        res = math_ops.matmul(arg, weights)
    return res

