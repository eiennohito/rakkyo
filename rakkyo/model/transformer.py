import tensorflow as tf
import numpy as np


class PositionalEmbedding(object):
    def __init__(self, num_units, max_length, scale=None):
        unit_size = num_units // 2
        vecs = np.arange(unit_size, dtype=np.float32) * 2 / unit_size
        vecs = np.float_power(10000, vecs)
        vecs = np.reshape(vecs, [1, unit_size])
        poss = np.reshape(np.arange(max_length), [max_length, 1])
        encodings_raw = poss / vecs
        encodings = np.zeros([max_length, unit_size * 2])
        encodings[:, 0::2] = np.sin(encodings_raw)
        encodings[:, 1::2] = np.cos(encodings_raw)
        if scale is not None:
            encodings *= np.reciprocal(np.sqrt(scale))

        self.pos_embed_table = tf.constant(encodings, dtype=tf.float32)

    def embeds(self, batch_maxlen, batch_size):
        position_ind = tf.tile(tf.expand_dims(tf.range(batch_maxlen), 0), [batch_size, 1])
        return tf.nn.embedding_lookup(self.pos_embed_table, position_ind)


def tf_activation(name):
    if name == 'selu':
        return tf.nn.selu
    elif name == 'relu':
        return tf.nn.relu
    elif name == 'lrelu':
        return tf.nn.leaky_relu
    elif name == 'softplus':
        return tf.nn.softplus
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'swish':
        return tf.nn.swish
    else:
        return None


def layer_norm(x: tf.Tensor, name=None):
    with tf.variable_scope(name, default_name='layer_norm'):
        shape = x.shape[-1]

        bias = tf.get_variable(
            name='bias',
            shape=[shape],
            initializer=tf.zeros_initializer()
        )

        gain = tf.get_variable(
            name='gain',
            shape=[shape],
            initializer=tf.ones_initializer()
        )

        mean, variance = tf.nn.moments(x, axes=-1, keep_dims=True)
        return tf.nn.batch_normalization(x, mean, variance, bias, gain, 1e-9)


def mlen(x: tf.Tensor):
    return tf.shape(x)[1]


class BaseSelfAttention(object):
    def __init__(self, iq, ik, iv, number_attn, inner_dim, key_dim, name=None, keep_prob=1.0,
                 key_bias=False, key_act=None, val_bias=False, val_act=None):
        with tf.variable_scope(name, default_name='attn'):
            self.input_shape = tf.shape(iv)
            self.batch_size = self.input_shape[0]

            self.Q = tf.layers.dense(
                inputs=iq,
                units=number_attn * key_dim,
                use_bias=key_bias, activation=tf_activation(key_act),
                name='Q'
            )

            self.K = tf.layers.dense(
                inputs=ik,
                units=number_attn * key_dim,
                use_bias=key_bias, activation=tf_activation(key_act),
                name='K'
            )

            self.V = tf.layers.dense(
                inputs=iv,
                units=number_attn * inner_dim,
                use_bias=val_bias, activation=tf_activation(val_act),
                name='V'
            )

            self.Qx = tf.reshape(self.Q, [self.batch_size, mlen(iq), number_attn, key_dim])
            self.Kx = tf.reshape(self.K, [self.batch_size, mlen(ik), number_attn, key_dim])
            self.Qx = tf.nn.dropout(self.Qx, keep_prob)
            self.Kx = tf.nn.dropout(self.Kx, keep_prob)

            self.AttnWeights = tf.einsum('bkhd,bqhd->bhkq', self.Kx, self.Qx)

            self.AttnWeights *= tf.rsqrt(tf.to_float(key_dim))

            self.Vx = tf.reshape(self.V, [self.batch_size, mlen(iv), number_attn, inner_dim])


class SelfAttention(BaseSelfAttention):
    def __init__(self, input, number_attn, inner_dim, key_dim, name=None, keep_prob=1.0, mask=None,
                 k_bias=False, k_act=None, v_bias=False, v_act=None, o_bias=False, o_act=None):
        super().__init__(input, input, input, number_attn, inner_dim, key_dim, name, keep_prob,
                         k_bias, k_act, v_bias, v_act)
        with tf.variable_scope(name, default_name='attn'):
            if mask is not None:
                self.AttnWeightsMasked = self.AttnWeights + mask
            else:
                self.AttnWeightsMasked = self.AttnWeights
            self.Attn = tf.nn.softmax(self.AttnWeightsMasked, axis=-1)

            self.Attended = tf.einsum('bhty,byhk->bthk', self.Attn, self.Vx)
            self.Ax = tf.reshape(self.Attended, [self.batch_size, mlen(input), number_attn * inner_dim])

            input_dim = input.shape[-1]

            self.output = tf.layers.dense(
                inputs=self.Ax,
                units=input_dim,
                use_bias=o_bias, activation=tf_activation(o_act),
                name='O'
            )


class FinalSelfAttention(BaseSelfAttention):
    def __init__(self, input, number_attn, inner_dim, key_dim, output, lens, name=None, keep_prob=1.0, out_bias=False,
                 out_act=None):
        super().__init__(input, input, input, number_attn, inner_dim, key_dim, name, keep_prob)
        with tf.variable_scope(name, default_name='attn'):
            self.AttnWeights = tf.reduce_mean(self.AttnWeights, axis=-1)

            self.LenMask = tf.sequence_mask(lens, mlen(input))
            self.Penalties = tf.to_float(tf.logical_not(self.LenMask)) * -1e5
            self.Penalties = tf.reshape(self.Penalties, [self.batch_size, 1, mlen(input)])

            self.Attn = tf.nn.softmax(self.AttnWeights + self.Penalties, axis=-1)

            self.LenModifier = tf.rsqrt(tf.to_float(lens))
            self.LenModifier = tf.reshape(self.LenModifier, [self.batch_size, 1, 1])
            # self.AttnScaled = self.Attn * self.LenModifier
            self.AttnScaled = self.Attn

            self.Attended = tf.einsum('bht,bthk->bhk', self.AttnScaled, self.Vx)
            self.Ax = tf.reshape(self.Attended, [self.batch_size, number_attn * inner_dim])

            self.output = tf.layers.dense(
                inputs=self.Ax,
                units=output,
                use_bias=out_bias, activation=out_act,
                name='O'
            )


class TransformerUnit(object):
    def __init__(self, input, number_attn, inner_dim, key_dim, proj_dim, name=None, attn_keep=1.0, mask=None,
                 normalize=True, keep_layer=1.0, activation='selu', oactivation='selu',
                 ak_bias=False, ak_act=None, av_bias=False, av_act=None, ao_bias=False, ao_act=None, gating=False):
        self.gating = gating
        with tf.variable_scope(name, default_name='tform'):
            input_dim = input.shape[-1]
            self.attn = SelfAttention(
                input=input,
                number_attn=number_attn,
                inner_dim=inner_dim,
                key_dim=key_dim,
                name='attn',
                keep_prob=attn_keep,
                mask=mask,
                k_bias=ak_bias,
                k_act=ak_act,
                v_bias=av_bias,
                v_act=av_act,
                o_bias=ao_bias,
                o_act=ao_act
            )
            self.inner = input + tf.nn.dropout(self.attn.output, keep_layer)
            if normalize:
                self.inner = layer_norm(self.inner, 'norm_attn')
            afunc = tf_activation(activation)
            oafunc = tf_activation(oactivation)
            self.tf_output = self.inner
            if proj_dim > 0:
                self.projected_inner = tf.layers.dense(self.tf_output, proj_dim, activation=afunc)
                self.tf_output = tf.layers.dense(self.projected_inner, input_dim, activation=oafunc)
            self.tf_output = tf.nn.dropout(self.tf_output, keep_layer)
            if gating:
                gate_inp = tf.concat([input, self.attn.output], axis=-1)
                self.gating_layer = tf.layers.Dense(
                    units=1,
                    dtype=input.dtype,
                    activation=tf.nn.sigmoid,
                    bias_initializer=tf.ones_initializer(input.dtype),
                    name='gate'
                )
                self.gate_values = self.gating_layer(gate_inp)
                self.tf_output = self.tf_output * self.gate_values
                self.output = input * (1 - self.gate_values) + self.tf_output
            else:
                self.output = self.inner + self.tf_output

            if normalize:
                self.output = layer_norm(self.output, 'norm_proj')
