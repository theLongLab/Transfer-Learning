###Simple CNN baseline
import inspect
from typing import Any, Callable, Dict, Optional, Text, Union, Iterable
import attention_module
import numpy as np
import sonnet as snt
import tensorflow as tf

# SEQUENCE_LENGTH = 196_608
# BIN_SIZE = 128
# TARGET_LENGTH = 896
class SimpleCNN(snt.Module):
    """
    Simple CNN baseline.
    Crop first → CNN → exact 896 bins.
    Input:  [B, 196608, 4]
    Output: [B, 896, 275]
    """
    def __init__(self, num_outputs=275, target_length=896, name="simple_cnn_196k_896"):
        super().__init__(name=name)
        self.target_length = target_length
        self.crop_length = target_length * 128  # 114,688 bp
        # -------------------------
        # Convolutional trunk
        # pooling: 2, 4, 4, 4
        # -------------------------
        self.trunk = snt.Sequential([
            # 114,688 → 57,344
            snt.Conv1D(128, kernel_shape=15, stride=2, padding="SAME"),
            tf.nn.relu,
            # 57,344 → 14,336
            snt.Conv1D(256, kernel_shape=15, stride=4, padding="SAME"),
            tf.nn.relu,
            # 14,336 → 3,584
            snt.Conv1D(512, kernel_shape=15, stride=4, padding="SAME"),
            tf.nn.relu,
            # 3,584 → 896
            snt.Conv1D(512, kernel_shape=15, stride=4, padding="SAME"),
            tf.nn.relu,
        ])
        # -------------------------
        # Output head
        # -------------------------
        self.head = snt.Sequential([
            snt.Conv1D(num_outputs, kernel_shape=1, padding="SAME"),
            tf.nn.softplus
        ])
    def _center_crop(self, x):
        """
        Crop input sequence before CNN.
        x: [B, 196608, 4]
        """
        L = tf.shape(x)[1]
        trim = (L - self.crop_length) // 2
        return x[:, trim:trim + self.crop_length, :]
    def __call__(self, x, is_training=False):
        x = self._center_crop(x)   # [B, 114688, 4]
        x = self.trunk(x)          # [B, 896, 512]
        x = self.head(x)           # [B, 896, 275]
        return {"human": x}



class TargetLengthCrop1D(snt.Module):
  """Crop sequence to match the desired target length."""

  def __init__(self, target_length: int, name='target_length_crop'):
    super().__init__(name=name)
    self._target_length = target_length

  def __call__(self, inputs):
    trim = (inputs.shape[-2] - self._target_length) // 2
    if trim < 0:
      raise ValueError('inputs longer than target length')

    return inputs[..., trim:-trim, :]


class Sequential(snt.Module):
  """snt.Sequential automatically passing is_training where it exists."""

  def __init__(self,
               layers: Optional[Union[Callable[[], Iterable[snt.Module]],
                                      Iterable[Callable[..., Any]]]] = None,
               name: Optional[Text] = None):
    super().__init__(name=name)
    if layers is None:
      self._layers = []
    else:
      # layers wrapped in a lambda function to have a common namespace.
      if hasattr(layers, '__call__'):
        with tf.name_scope(name):
          layers = layers()
      self._layers = [layer for layer in layers if layer is not None]

  def __call__(self, inputs: tf.Tensor, is_training: bool, **kwargs):
    outputs = inputs
    for _, mod in enumerate(self._layers):
      if accepts_is_training(mod):
        outputs = mod(outputs, is_training=is_training, **kwargs)
      else:
        outputs = mod(outputs, **kwargs)
    return outputs


def pooling_module(kind, pool_size):
  """Pooling module wrapper."""
  if kind == 'attention':
    return SoftmaxPooling1D(pool_size=pool_size, per_channel=True,
                            w_init_scale=2.0)
  elif kind == 'max':
    return tf.keras.layers.MaxPool1D(pool_size=pool_size, padding='same')
  else:
    raise ValueError(f'Invalid pooling kind: {kind}.')


class SoftmaxPooling1D(snt.Module):
  """Pooling operation with optional weights."""

  def __init__(self,
               pool_size: int = 2,
               per_channel: bool = False,
               w_init_scale: float = 0.0,
               name: str = 'softmax_pooling'):
    """Softmax pooling.

    Args:
      pool_size: Pooling size, same as in Max/AvgPooling.
      per_channel: If True, the logits/softmax weights will be computed for
        each channel separately. If False, same weights will be used across all
        channels.
      w_init_scale: When 0.0 is equivalent to avg pooling, and when
        ~2.0 and `per_channel=False` it's equivalent to max pooling.
      name: Module name.
    """
    super().__init__(name=name)
    self._pool_size = pool_size
    self._per_channel = per_channel
    self._w_init_scale = w_init_scale
    self._logit_linear = None

  @snt.once
  def _initialize(self, num_features):
    self._logit_linear = snt.Linear(
        output_size=num_features if self._per_channel else 1,
        with_bias=False,  # Softmax is agnostic to shifts.
        w_init=snt.initializers.Identity(self._w_init_scale))

  def __call__(self, inputs):
    _, length, num_features = inputs.shape
    self._initialize(num_features)
    inputs = tf.reshape(
        inputs,
        (-1, length // self._pool_size, self._pool_size, num_features))
    return tf.reduce_sum(
        inputs * tf.nn.softmax(self._logit_linear(inputs), axis=-2),
        axis=-2)


class Residual(snt.Module):
  """Residual block."""

  def __init__(self, module: snt.Module, name='residual'):
    super().__init__(name=name)
    self._module = module

  def __call__(self, inputs: tf.Tensor, is_training: bool, *args,
               **kwargs) -> tf.Tensor:
    return inputs + self._module(inputs, is_training, *args, **kwargs)


def gelu(x: tf.Tensor) -> tf.Tensor:
  """Applies the Gaussian error linear unit (GELU) activation function.

  Using approximiation in section 2 of the original paper:
  https://arxiv.org/abs/1606.08415

  Args:
    x: Input tensor to apply gelu activation.
  Returns:
    Tensor with gelu activation applied to it.
  """
  return tf.nn.sigmoid(1.702 * x) * x


def one_hot_encode(sequence: str,
                   alphabet: str = 'ACGT',
                   neutral_alphabet: str = 'N',
                   neutral_value: Any = 0,
                   dtype=np.float32) -> np.ndarray:
  """One-hot encode sequence."""
  def to_uint8(string):
    return np.frombuffer(string.encode('ascii'), dtype=np.uint8)
  hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
  hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
  hash_table[to_uint8(neutral_alphabet)] = neutral_value
  hash_table = hash_table.astype(dtype)
  return hash_table[to_uint8(sequence)]


def exponential_linspace_int(start, end, num, divisible_by=1):
  """Exponentially increasing values of integers."""
  def _round(x):
    return int(np.round(x / divisible_by) * divisible_by)

  base = np.exp(np.log(end / start) / (num - 1))
  return [_round(start * base**i) for i in range(num)]


def accepts_is_training(module):
  return 'is_training' in list(inspect.signature(module.__call__).parameters)
