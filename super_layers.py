#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

### NOTE: the API of this file is based on Paddle2.0, the API in layers_old.py is based on Paddle1.8

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid.core as core

# from ...common import get_logger
from utils import compute_start_end, get_same_padding, convert_to_list

__all__ = [
    'SuperConv2D', 'SuperBatchNorm2D', 'SuperLinear', 'Block',
]

### TODO: if task is elastic width, need to add re_organize_middle_weight in 1x1 conv in MBBlock

_cnt = 0


def counter():
    global _cnt
    _cnt += 1
    return _cnt


class BaseBlock(paddle.nn.Layer):
    def __init__(self, key=None):
        super(BaseBlock, self).__init__()
        if key is not None:
            self._key = str(key)
        else:
            self._key = self.__class__.__name__ + str(counter())

    # set SuperNet class
    def set_supernet(self, supernet):
        self.__dict__['supernet'] = supernet

    @property
    def key(self):
        return self._key


class Block(BaseBlock):
    """
    Model is composed of nest blocks.

    Parameters:
        fn(paddle.nn.Layer): instance of super layers, such as: SuperConv2D(3, 5, 3).
        fixed(bool, optional): whether to fix the shape of the weight in this layer. Default: False.
        key(str, optional): key of this layer, one-to-one correspondence between key and candidate config. Default: None.
    """

    def __init__(self, fn, fixed=False, key=None):
        super(Block, self).__init__(key)
        self.fn = fn
        self.fixed = fixed
        self.candidate_config = self.fn.candidate_config

    def forward(self, *inputs, **kwargs):
        out = self.supernet.layers_forward(self, *inputs, **kwargs)
        return out


class SuperConv2D(nn.Conv2D):
    """
    This interface is used to construct a callable object of the ``SuperConv2D``  class.

    Note: the channel in config need to less than first defined.

    The super convolution2D layer calculates the output based on the input, filter
    and strides, paddings, dilations, groups parameters. Input and
    Output are in NCHW format, where N is batch size, C is the number of
    the feature map, H is the height of the feature map, and W is the width of the feature map.
    Filter's shape is [MCHW] , where M is the number of output feature map,
    C is the number of input feature map, H is the height of the filter,
    and W is the width of the filter. If the groups is greater than 1,
    C will equal the number of input feature map divided by the groups.
    Please refer to UFLDL's `convolution
    <http://ufldl.stanford.edu/tutorial/supervised/FeatureExtractionUsingConvolution/>`_
    for more details.
    If bias attribution and activation type are provided, bias is added to the
    output of the convolution, and the corresponding activation function is
    applied to the final result.
    For each input :math:`X`, the equation is:
    .. math::
        Out = \\sigma (W \\ast X + b)
    Where:
    * :math:`X`: Input value, a ``Tensor`` with NCHW format.
    * :math:`W`: Filter value, a ``Tensor`` with shape [MCHW] .
    * :math:`\\ast`: Convolution operation.
    * :math:`b`: Bias value, a 2-D ``Tensor`` with shape [M, 1].
    * :math:`\\sigma`: Activation function.
    * :math:`Out`: Output value, the shape of :math:`Out` and :math:`X` may be different.

    Example:
        - Input:
          Input shape: :math:`(N, C_{in}, H_{in}, W_{in})`
          Filter shape: :math:`(C_{out}, C_{in}, H_f, W_f)`
        - Output:
          Output shape: :math:`(N, C_{out}, H_{out}, W_{out})`
        Where
        .. math::
            H_{out}&= \\frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]} + 1 \\\\
            W_{out}&= \\frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]} + 1
    Parameters:
        num_channels(int): The number of channels in the input image.
        num_filters(int): The number of filter. It is as same as the output
            feature map.
        filter_size (int or tuple): The filter size. If filter_size is a tuple,
            it must contain two integers, (filter_size_H, filter_size_W).
            Otherwise, the filter will be a square.
        candidate_config(dict, optional): Dictionary descripts candidate config of this layer,
            such as {'kernel_size': (3, 5, 7), 'channel': (4, 6, 8)}, means the kernel size of 
            this layer can be choose from (3, 5, 7), the key of candidate_config
            only can be 'kernel_size', 'channel' and 'expand_ratio', 'channel' and 'expand_ratio'
            CANNOT be set at the same time. Default: None.
        transform_kernel(bool, optional): Whether to use transform matrix to transform a large filter
            to a small filter. Default: False.
        stride (int or tuple, optional): The stride size. If stride is a tuple, it must
            contain two integers, (stride_H, stride_W). Otherwise, the
            stride_H = stride_W = stride. Default: 1.
        padding (int or tuple, optional): The padding size. If padding is a tuple, it must
            contain two integers, (padding_H, padding_W). Otherwise, the
            padding_H = padding_W = padding. Default: 0.
        dilation (int or tuple, optional): The dilation size. If dilation is a tuple, it must
            contain two integers, (dilation_H, dilation_W). Otherwise, the
            dilation_H = dilation_W = dilation. Default: 1.
        groups (int, optional): The groups number of the Conv2d Layer. According to grouped
            convolution in Alex Krizhevsky's Deep CNN paper: when group=2,
            the first half of the filters is only connected to the first half
            of the input channels, while the second half of the filters is only
            connected to the second half of the input channels. Default: 1.
        param_attr (ParamAttr, optional): The parameter attribute for learnable weights(Parameter)
            of conv2d. If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as param_attr. If the Initializer of the param_attr
            is not set, the parameter is initialized with :math:`Normal(0.0, std)`,
            and the :math:`std` is :math:`(\\frac{2.0 }{filter\_elem\_num})^{0.5}`. Default: None.
        bias_attr (ParamAttr or bool, optional): The attribute for the bias of conv2d.
            If it is set to False, no bias will be added to the output units.
            If it is set to None or one attribute of ParamAttr, conv2d
            will create ParamAttr as bias_attr. If the Initializer of the bias_attr
            is not set, the bias is initialized zero. Default: None.
        use_cudnn (bool, optional): Use cudnn kernel or not, it is valid only when the cudnn
            library is installed. Default: True.
        act (str, optional): Activation type, if it is set to None, activation is not appended.
            Default: None.
        dtype (str, optional): Data type, it can be "float32" or "float64". Default: "float32".
    Attribute:
        **weight** (Parameter): the learnable weights of filter of this layer.
        **bias** (Parameter or None): the learnable bias of this layer.
    Returns:
        None
    
    Raises:
        ValueError: if ``use_cudnn`` is not a bool value.
    Examples:
        .. code-block:: python
          import paddle 
          from paddleslim.nas.ofa.layers import SuperConv2D
          import numpy as np
          data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
          super_conv2d = SuperConv2D(3, 10, 3)
          config = {'channel': 5}
          data = paddle.to_variable(data)
          conv = super_conv2d(data, config)

    """

    ### NOTE: filter_size, num_channels and num_filters must be the max of candidate to define a largest network.
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 candidate_config={},
                 transform_kernel=False,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW'):
        super(SuperConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            dilation=dilation,
            groups=groups,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            data_format=data_format)

        self.candidate_config = candidate_config
        if len(candidate_config.items()) != 0:
            for k, v in candidate_config.items():
                candidate_config[k] = list(set(v))

        self.ks_set = candidate_config[
            'kernel_size'] if 'kernel_size' in candidate_config else None

        self.expand_ratio = candidate_config[
            'expand_ratio'] if 'expand_ratio' in candidate_config else None
        self.channel = candidate_config[
            'channel'] if 'channel' in candidate_config else None
        self.base_channel = self._out_channels
        if self.expand_ratio != None:
            self.base_channel = int(self._out_channels / max(self.expand_ratio))

        self.transform_kernel = transform_kernel
        if self.ks_set != None:
            self.ks_set.sort()
        if self.transform_kernel != False:
            scale_param = dict()
            ### create parameter to transform kernel
            for i in range(len(self.ks_set) - 1):
                ks_small = self.ks_set[i]
                ks_large = self.ks_set[i + 1]
                param_name = '%dto%d_matrix' % (ks_large, ks_small)
                ks_t = ks_small**2
                scale_param[param_name] = self.create_parameter(
                    attr=paddle.ParamAttr(
                        name=self._full_name + param_name,
                        initializer=nn.initializer.Assign(np.eye(ks_t))),
                    shape=(ks_t, ks_t),
                    dtype=self._dtype)

            for name, param in scale_param.items():
                setattr(self, name, param)

        self.cur_out_channels = None
        self.cur_in_channels = None

    def get_active_filter(self, in_nc, out_nc, kernel_size):
        start, end = compute_start_end(self._kernel_size[0], kernel_size)
        ### if NOT transform kernel, intercept a center filter with kernel_size from largest filter

        filters = self.weight[:out_nc, :in_nc, start:end, start:end]

        if self.transform_kernel != False and kernel_size < self._kernel_size[
                0]:
            ### if transform kernel, then use matrix to transform
            start_filter = self.weight[:out_nc, :in_nc, :, :]
            for i in range(len(self.ks_set) - 1, 0, -1):
                src_ks = self.ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self.ks_set[i - 1]
                start, end = compute_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = paddle.reshape(
                    _input_filter,
                    shape=[(_input_filter.shape[0] * _input_filter.shape[1]),
                           -1])
                _input_filter = paddle.matmul(
                    _input_filter,
                    self.__getattr__('%dto%d_matrix' %
                                     (src_ks, target_ks)), False, False)
                _input_filter = paddle.reshape(
                    _input_filter,
                    shape=[
                        filters.shape[0], filters.shape[1], target_ks, target_ks
                    ])
                start_filter = _input_filter
            filters = start_filter
        return filters

    def get_groups_in_out_nc(self, in_nc, out_nc):
        if self._groups == 1:
            ### standard conv
            return self._groups, in_nc, out_nc
        elif self._groups == self._in_channels:
            ### depthwise convolution
            # if in_nc != out_nc:
                # _logger.debug(
                #     "input channel and output channel in depthwise conv is different, change output channel to input channel! origin channel:(in_nc {}, out_nc {}): ".
                #     format(in_nc, out_nc))
            groups = in_nc
            out_nc = in_nc
            return groups, in_nc, out_nc
        else:
            ### groups convolution
            ### conv: weight: (Cout, Cin/G, Kh, Kw)
            groups = self._groups
            in_nc = int(in_nc // groups)
            return groups, in_nc, out_nc

    def forward(self, input, kernel_size=None, expand_ratio=None, channel=None):
        """
        Parameters:
            input(Tensor): Input tensor.
            kernel_size(int, optional): the kernel size of the filter in actual calculation. Default: None.
            expand_ratio(int|float, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
            channel(int, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
        """
        self.cur_config = {
            'kernel_size': kernel_size,
            'expand_ratio': expand_ratio,
            'channel': channel
        }
        in_nc = int(input.shape[1])
        # assert (
        #     expand_ratio == None or channel == None
        # ), "expand_ratio and channel CANNOT be NOT None at the same time."
        # if expand_ratio != None:
        #     out_nc = int(expand_ratio * self.base_channel)
        # elif channel != None:
        #     out_nc = int(channel)
        # else:
        #     out_nc = self._out_channels
        ks = int(self._kernel_size[0]) if kernel_size == None else int(
            kernel_size)

        out_nc = self.cur_out_channels

        groups, weight_in_nc, weight_out_nc = self.get_groups_in_out_nc(in_nc,
                                                                        out_nc)

        weight = self.get_active_filter(weight_in_nc, weight_out_nc, ks)

        if kernel_size != None or 'kernel_size' in self.candidate_config.keys():
            padding = convert_to_list(get_same_padding(ks), 2)
        else:
            padding = self._padding

        if self.bias is not None:
            bias = self.bias[:out_nc]
        else:
            bias = self.bias

        out = F.conv2d(
            input,
            weight,
            bias=bias,
            stride=self._stride,
            padding=padding,
            dilation=self._dilation,
            groups=self._groups,
            data_format=self._data_format)
        return out


class SuperLinear(nn.Linear):
    """
    Super Fully-connected linear transformation layer. 
    
    For each input :math:`X` , the equation is:
    .. math::
        Out = XW + b
    where :math:`W` is the weight and :math:`b` is the bias.
    Linear layer takes only one multi-dimensional tensor as input with the
    shape :math:`[batch\_size, *, in\_features]` , where :math:`*` means any
    number of additional dimensions. It multiplies input tensor with the weight
    (a 2-D tensor of shape :math:`[in\_features, out\_features]` ) and produces
    an output tensor of shape :math:`[batch\_size, *, out\_features]` .
    If :math:`bias\_attr` is not False, the bias (a 1-D tensor of
    shape :math:`[out\_features]` ) will be created and added to the output.
    Parameters:
        in_features (int): The number of input units.
        out_features (int): The number of output units.
        candidate_config(dict, optional): Dictionary descripts candidate config of this layer,
            such as {'channel': (4, 6, 8)}, the key of candidate_config
            only can be 'channel' and 'expand_ratio', 'channel' and 'expand_ratio'
            CANNOT be set at the same time. Default: None.
        weight_attr (ParamAttr, optional): The attribute for the learnable
            weight of this layer. The default value is None and the weight will be
            initialized to zero. For detailed information, please refer to
            paddle.ParamAttr.
        bias_attr (ParamAttr|bool, optional): The attribute for the learnable bias
            of this layer. If it is set to False, no bias will be added to the output.
            If it is set to None or one kind of ParamAttr, a bias parameter will
            be created according to ParamAttr. For detailed information, please refer
            to paddle.ParamAttr. The default value is None and the bias will be
            initialized to zero.
        name (str, optional): Normally there is no need for user to set this parameter.
            For detailed information, please refer to :ref:`api_guide_Name` .
    Attribute:
        **weight** (Parameter): the learnable weight of this layer.
        **bias** (Parameter): the learnable bias of this layer.
    Shape:
        - input: Multi-dimentional tensor with shape :math:`[batch\_size, *, in\_features]` .
        - output: Multi-dimentional tensor with shape :math:`[batch\_size, *, out\_features]` .
    Examples:
        .. code-block:: python
          import numpy as np
          import paddle
          from paddleslim.nas.ofa.layers import SuperLinear
          
          data = np.random.uniform(-1, 1, [32, 64] ).astype('float32')
          config = {'channel': 16}
          linear = SuperLinear(32, 64)
          data = paddle.to_variable(data)
          res = linear(data, **config)
    """

    def __init__(self,
                 in_features,
                 out_features,
                 candidate_config={},
                 weight_attr=None,
                 bias_attr=None,
                 name=None):
        super(SuperLinear, self).__init__(in_features, out_features,
                                          weight_attr, bias_attr, name)
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self._in_features = in_features
        self._out_features = out_features
        self.candidate_config = candidate_config
        self.expand_ratio = candidate_config[
            'expand_ratio'] if 'expand_ratio' in candidate_config else None
        self.base_output_dim = self._out_features
        if self.expand_ratio != None:
            self.base_output_dim = int(self._out_features /
                                       max(self.expand_ratio))

        self.cur_out_features = None
        self.cur_in_features = None

    def forward(self, input, expand_ratio=None, channel=None):
        """
        Parameters:
            input(Tensor): input tensor.
            expand_ratio(int|float, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
            channel(int, optional): the expansion ratio of filter's channel number in actual calculation. Default: None.
        """
        self.cur_config = {'expand_ratio': expand_ratio, 'channel': channel}
        ### weight: (Cin, Cout)
        in_nc = int(input.shape[-1])
        # assert (
        #     expand_ratio == None or channel == None
        # ), "expand_ratio and channel CANNOT be NOT None at the same time."
        # if expand_ratio != None:
        #     out_nc = int(expand_ratio * self.base_output_dim)
        # elif channel != None:
        #     out_nc = int(channel)
        # else:
        #     out_nc = self._out_features

        out_nc = self.cur_out_features

        weight = self.weight[:in_nc, :out_nc]
        if self._bias_attr != False:
            bias = self.bias[:out_nc]
        else:
            bias = self.bias

        out = F.linear(x=input, weight=weight, bias=bias, name=self.name)
        return out


# class SuperBatchNorm2D(nn.BatchNorm2D):
#     """
#     This interface is used to construct a callable object of the ``SuperBatchNorm2D`` class. 

#     Parameters:
#         num_features(int): Indicate the number of channels of the input ``Tensor``.
#         epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
#         momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
#         weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
#             of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
#             will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
#             If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
#         bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
#             If it is set to None or one attribute of ParamAttr, batch_norm
#             will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
#             If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
#         data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
#         name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..

#     Examples:
#        .. code-block:: python
#          import paddle
#          import numpy as np
#          from paddleslim.nas.ofa.layers import SuperBatchNorm2D
         
#          np.random.seed(123)
#          x_data = np.random.random(size=(2, 5, 2, 3)).astype('float32')
#          x = paddle.to_tensor(x_data)
#          batch_norm = SuperBatchNorm2D(5)
#          batch_norm_out = batch_norm(x)
#     """

#     def __init__(self,
#                  num_features,
#                  momentum=0.9,
#                  epsilon=1e-05,
#                  weight_attr=None,
#                  bias_attr=None,
#                  data_format='NCHW',
#                  name=None):
#         super(SuperBatchNorm2D, self).__init__(num_features, momentum, epsilon,
#                                                weight_attr, bias_attr,
#                                                data_format, name)

#     def forward(self, input):
#         self._check_data_format(self._data_format)
#         self._check_input_dim(input)

#         feature_dim = int(input.shape[1])

#         weight = self.weight[:feature_dim]
#         bias = self.bias[:feature_dim]
#         mean = self._mean[:feature_dim]
#         variance = self._variance[:feature_dim]

#         return F.batch_norm(
#             input,
#             mean,
#             variance,
#             weight=weight,
#             bias=bias,
#             training=self.training,
#             momentum=self._momentum,
#             epsilon=self._epsilon,
#             data_format=self._data_format)

import paddle.fluid.core as core
class SuperBatchNorm2D(nn.BatchNorm2D):
    """
    This interface is used to construct a callable object of the ``SuperBatchNorm2D`` class. 
    Parameters:
        num_features(int): Indicate the number of channels of the input ``Tensor``.
        epsilon(float, optional): The small value added to the variance to prevent division by zero. Default: 1e-5.
        momentum(float, optional): The value used for the moving_mean and moving_var computation. Default: 0.9.
        weight_attr(ParamAttr|bool, optional): The parameter attribute for Parameter `scale`
            of batch_norm. If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as weight_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the weight_attr is not set, the parameter is initialized with Xavier. Default: None.
        bias_attr(ParamAttr|bool, optional): The parameter attribute for the bias of batch_norm.
            If it is set to None or one attribute of ParamAttr, batch_norm
            will create ParamAttr as bias_attr. If it is set to Fasle, the weight is not learnable.
            If the Initializer of the bias_attr is not set, the bias is initialized zero. Default: None.
        data_format(str, optional): Specify the input data format, the data format can be "NCHW" or "NHWC". Default: NCHW.
        name(str, optional): Name for the BatchNorm, default is None. For more information, please refer to :ref:`api_guide_Name`..
    Examples:
       .. code-block:: python
         import paddle
         import numpy as np
         from paddleslim.nas.ofa.layers import SuperBatchNorm2D
         
         np.random.seed(123)
         x_data = np.random.random(size=(2, 5, 2, 3)).astype('float32')
         x = paddle.to_tensor(x_data)
         batch_norm = SuperBatchNorm2D(5)
         batch_norm_out = batch_norm(x)
    """

    def __init__(self,
                 num_features,
                 momentum=0.9,
                 epsilon=1e-05,
                 weight_attr=None,
                 bias_attr=None,
                 data_format='NCHW',
                 use_global_stats=None,
                 name=None):
        super(SuperBatchNorm2D, self).__init__(
            num_features, momentum, epsilon, weight_attr, bias_attr,
            data_format, use_global_stats, name)

    def forward(self, input):
        self._check_data_format(self._data_format)
        self._check_input_dim(input)

        feature_dim = int(input.shape[1])

        weight = self.weight[:feature_dim]
        bias = self.bias[:feature_dim]
        mean = self._mean[:feature_dim]
        variance = self._variance[:feature_dim]

        mean_out = self._mean
        variance_out = self._variance
        mean_out_tmp = mean
        variance_out_tmp = variance

        if self._use_global_stats == None:
            # self._use_global_stats = not self.training
            use_global_stats = not self.training
            trainable_statistics = False
        else:
            trainable_statistics = not self._use_global_stats

        attrs = ("momentum", self._momentum, "epsilon", self._epsilon, "is_test",
                 not self.training, "data_layout", self._data_format, "use_mkldnn", False,
                 "fuse_with_relu", False, "use_global_stats", use_global_stats,
                 "trainable_statistics", trainable_statistics)

        if feature_dim != self._mean.shape[0]:
            batch_norm_out = core.ops.batch_norm(input, weight, bias, mean,
                                                 variance, mean_out_tmp,
                                                 variance_out_tmp, *attrs)
            self._mean[:feature_dim] = mean
            self._variance[:feature_dim] = variance
            mean_out[:feature_dim] = mean_out_tmp
            variance_out[:feature_dim] = variance_out_tmp
        else:
            batch_norm_out = core.ops.batch_norm(input, weight, bias,
                                                 self._mean, self._variance,
                                                 mean_out, variance_out, *attrs)

        return batch_norm_out[0]
