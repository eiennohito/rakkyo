import model.transformer as trf
from pyhocon import ConfigTree as Config
import tensorflow as tf

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import MorphModel


class Layer(object):
    def __init__(self, cfg: Config, model: 'MorphModel', input, name):
        self.cfg = cfg
        self.input = input
        self.model = model
        self.name = name
        self.debug = cfg.get_bool('debug', model.debug)
        self.length_handler = model.len_handler
        self.concat_handled = False

    def dropout_prob(self, model, cfg, name):
        if model.infer:
            return 1.0
        return cfg.get_float(name, 1.0)


class TransformerLayer(Layer):
    def __init__(self, cfg: Config, model, input, name):
        super().__init__(cfg, model, input, name)
        tagging = isinstance(input, list)

        if tagging:
            self.output = []

            for i, x in enumerate(input):
                impl = self.make_transformer(cfg, f't{i}', model)
                self.output.append(impl.output)

            tagwise = cfg.get_int('tagwise', None)

            if tagwise is not None:
                xinput = tf.concat(self.output, axis=-1)

                tagwise_act = cfg.get_string('tagwise_activation', 'selu')
                tagwise_bias = cfg.get_bool('tagwise_bias', True)
                tagwise_act = trf.tf_activation(tagwise_act)
                self.tagwise_in = tf.layers.dense(
                    inputs=xinput,
                    units=tagwise,
                    activation=tagwise_act,
                    use_bias=tagwise_bias,
                    name='tagwise_in'
                )

                tagwise_dims = [x.shape[-1] for x in input]
                tagwise_alldims = sum(tagwise_dims)
                self.tagwise_out = tf.layers.dense(
                    inputs=self.tagwise_in,
                    units=tagwise_alldims,
                    activation=tagwise_act,
                    use_bias=tagwise_bias,
                    name='tagwise_out'
                )

                tagwise_splitted = tf.split(self.tagwise_out, tagwise_dims, axis=2)

                self.output = [(x + y) for x, y in zip(self.output, tagwise_splitted)]
        else:
            self.impl = self.make_transformer(cfg, name, model, input)
            self.output = self.impl.output

    def make_transformer(self, cfg: Config, name, model, input):
        d = model.data
        dim = [d.batch_size, 1, 1, self.length_handler.batch_mlen]
        mask = tf.reshape(self.length_handler.length_softmax_mask, dim)
        impl = trf.TransformerUnit(
            input=input,
            number_attn=(cfg.get_int('num_heads', 8)),
            inner_dim=cfg.get_int('inner', 16),
            key_dim=cfg.get_int('key', 16),
            proj_dim=cfg.get_int('proj', 16),
            attn_keep=self.dropout_prob(model, cfg, 'attn_keep'),
            mask=mask,
            normalize=cfg.get_bool('normalize', True),
            keep_layer=self.dropout_prob(model, cfg, 'keep_layer'),
            activation=cfg.get_string('activation', 'selu'),
            oactivation=cfg.get_string('oactivation', 'selu'),
            ak_bias=cfg.get_bool('ak_bias', False),
            ak_act=cfg.get_string('ak_act', None),
            av_bias=cfg.get_bool('av_bias', False),
            av_act=cfg.get_string('av_act', None),
            ao_bias=cfg.get_bool('ao_bias', False),
            ao_act=cfg.get_string('ao_act', None),
            gating=cfg.get_bool('gate', False)
        )

        if self.debug:
            import input.imager as imgr
            imger = imgr.Imager()
            tf.summary.histogram(f'raw_attn', impl.attn.AttnWeights, family=name)
            num_heads = cfg.get_int('num_heads', 8)
            attn0 = impl.attn.Attn[0]
            if impl.gating:
                tf.summary.histogram('gate', impl.gate_values, family=name)
                gate0 = impl.gate_values[0]
                mlen = trf.mlen(input)
                gate0x = tf.reshape(gate0, [1, mlen, 1])
                gate0x = tf.tile(gate0x, [num_heads, 1, 1])
                attn0 = tf.concat([gate0x, attn0], axis=-1)
            tf.summary.image(f'attn', imger.to_image_norm(attn0), max_outputs=(
                num_heads), family=name)

        return impl


class DenseLayer(Layer):
    def __init__(self, cfg: Config, model, input, name):
        super().__init__(cfg, model, input, name)
        split = cfg.get_bool('split', False)
        size = cfg.get_int('size', input.shape[-1].value)

        if split:
            size *= model.num_taggers

        self.impl = tf.layers.Dense(
            units=size,
            activation=trf.tf_activation(cfg.get_string('activation', 'selu')),
            use_bias=cfg.get_bool('bias', True),
            name=name
        )

        self.output = self.impl(input)

        if split:
            self.output = tf.split(self.output, model.num_taggers, axis=2)


class DoubleLayer(Layer):
    def __init__(self, cfg: Config, model, input, name):
        super().__init__(cfg, model, input, name)
        self.output = tf.concat([input, input], axis=-1)
        if cfg.get_bool('keep_length', False):
            self.output *= 2.0 ** -0.5


class RnnLayer(Layer):
    def __init__(self, cfg: Config, model, input, name):
        super().__init__(cfg, model, input, name)
        self.size = cfg.get_int('size', input.shape[-1].value)
        self.bidi = cfg.get_bool('bidi', True)

        if self.bidi:
            cell_size = self.size // 2

            self.rnn_out, self.rnn_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.cell(cell_size + self.size % 2, "fwd"),
                cell_bw=self.cell(cell_size, "bwd"),
                inputs=input,
                sequence_length=self.length_handler.length,
                swap_memory=True,
                time_major=False,
                dtype=input.dtype,
                scope=tf.get_variable_scope()
            )

            self.output = tf.concat(self.rnn_out, 2)
        else:
            self.rnn_out, self.rnn_state = tf.nn.dynamic_rnn(
                cell=self.cell(self.size),
                inputs=input,
                sequence_length=self.length_handler.length,
                swap_memory=True,
                time_major=False,
                dtype=input.dtype,
                scope=tf.get_variable_scope()
            )

            self.output = self.rnn_out

        if cfg.get_bool('residual', False):
            insize = input.shape[-1].value

            if insize != self.size:
                projected = tf.layers.dense(
                    inputs=self.output,
                    units=insize,
                    activation=tf.nn.selu
                )
            else:
                projected = self.output

            self.output = input + projected

    def cell(self, size, name='cell'):
        ctype = self.cfg.get_string('cell', 'lstm')

        if ctype == 'lstm':
            return tf.contrib.rnn.LSTMBlockCell(
                num_units=size,
                use_peephole=self.cfg.get_bool('lstm_peephole', True),
                name=name
            )
        elif ctype == 'gru':
            return tf.contrib.rnn.GRUBlockCell(
                num_units=size,
                name=name
            )
        elif ctype == 'lnlstm':
            return tf.contrib.rnn.LayerNormBasicLSTMCell(
                num_units=size,
                layer_norm=self.cfg.get_bool('use_norm', True),
                dropout_keep_prob=self.dropout_prob(self.model, self.cfg, 'keep_layer')
            )
        else:
            raise Exception("Unknown cell type:", ctype)


class ConvLayer(Layer):
    def __init__(self, cfg: Config, model, input, name):
        super().__init__(cfg, model, input, name)
        widths = cfg.get_list('widths')
        sizes = cfg.get_list('sizes')

        out_size = sum(sizes)

        assert out_size > 0
        d = model.data

        reshaped_input = tf.reshape(input, [d.batch_size, self.length_handler.batch_mlen, 1, -1])
        idim = input.shape[-1].value

        outs = []

        for w, s, i in zip(widths, sizes, range(len(sizes))):
            assert w % 2 == 1

            filter = tf.get_variable(
                name=f'filter_{i}',
                shape=[w, 1, idim, s],
                dtype=input.dtype,
                initializer=tf.random_normal_initializer(
                    dtype=input.dtype
                )
            )

            npad = w // 2
            xinput = tf.pad(reshaped_input, [[0, 0], [npad, npad], [0, 0], [0, 0]])

            xout = tf.nn.conv2d(
                input=xinput,
                filter=filter,
                strides=[1, 1, 1, 1],
                padding='VALID',
                name=f'conv_{i}'
            )

            xout = tf.reshape(xout, [d.batch_size, self.length_handler.batch_mlen, s])

            outs.append(xout)

        if len(outs) == 1:
            self.output = outs[0]
        else:
            self.output = tf.concat(outs, axis=2)

        act = trf.tf_activation(cfg.get_string('activation', None))
        if act is not None:
            self.output = act(self.output)

        with tf.variable_scope(name):
            if cfg.get_bool('residual', False):
                if idim != out_size:
                    projected = tf.layers.dense(
                        inputs=self.output,
                        units=idim,
                        activation=tf.nn.selu
                    )
                else:
                    projected = self.output

                self.output = input + projected


class PosEmbeddingLayer(Layer):
    def __init__(self, cfg: Config, model, input, name):
        super().__init__(cfg, model, input, name)
        size = cfg.get_int('size', None)

        if size is None:
            size = input.shape[-1].value

            self.pemb = trf.PositionalEmbedding(size, model.max_length)
            embeds = self.pemb.embeds(self.length_handler.batch_mlen, model.data.batch_size)
            self.output = input + embeds
        else:
            cur_size = input.shape[-1].value
            self.pemb = trf.PositionalEmbedding(size, model.max_length)
            embeds = self.pemb.embeds(self.length_handler.batch_mlen, model.data.batch_size)
            pad = size - cur_size
            if pad < 0:
                raise Exception("Invalid positional embedding size:", name, size, cur_size)
            if pad > 0:
                input = tf.pad(input, [[0, 0], [0, 0], [pad, 0]])
            self.output = input + embeds


class NoopLayer(Layer):
    def __init__(self, cfg: Config, model, input, name):
        super().__init__(cfg, model, input, name)
        self.output = input


class Layers(object):
    registry = {
        'tform': TransformerLayer,
        'dense': DenseLayer,
        'double': DoubleLayer,
        'rnn': RnnLayer,
        'pemb': PosEmbeddingLayer,
        'conv': ConvLayer,
        'noop': NoopLayer,
    }

    @staticmethod
    def for_name(name):
        return Layers.registry[name]
