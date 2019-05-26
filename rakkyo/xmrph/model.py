import typing
import tensorflow as tf
from pyhocon import ConfigTree as Config

from .layers import Layers
from .input import Dictionaries, InputData
import model.transformer as trf
import sys
import numpy as np


class BertObjective(object):
    def __init__(self, data: InputData, cfg: Config, model: 'MorphModel'):

        self.weight = cfg.get_float('bert.weight', 1)

        self.bert_bias = tf.get_variable(
            name='bert_bias',
            shape=model.char_dict.size,
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )

        bert_raw = data.next["bert"]
        chars_orig = data.next["chars_orig"]
        bert_mask = tf.greater(bert_raw, 0)

        layer_no = cfg.get_int('bert.layer', -1)
        last_layer = model.layers[layer_no]

        queries = tf.boolean_mask(last_layer.output, bert_mask)
        qdim_size = queries.shape[-1].value
        if qdim_size != model.char_emb_size:
            queries = tf.layers.dense(
                queries,
                model.char_emb_size,
                name="bert_out_proj"
            )

        self.queries = queries

        labels = tf.boolean_mask(chars_orig, bert_mask)

        nsampled = cfg.get_int('bert.sample', 50)
        if nsampled > 0:
            labels = tf.reshape(labels, [-1, 1])
            labels = tf.cast(labels, tf.int64)

            sampled = tf.nn.learned_unigram_candidate_sampler(
                true_classes=labels,
                num_true=1,
                num_sampled=nsampled,
                unique=False,
                range_max=model.char_dict.size,
                name="bert_sample"
            )

            self.raw_loss = tf.nn.sampled_softmax_loss(
                weights=model.char_embeddings,
                biases=self.bert_bias,
                labels=labels,
                inputs=queries,
                num_sampled=nsampled,
                num_classes=model.char_dict.size,
                sampled_values=sampled,
                name="bert_loss"
            )
        else:
            logits = tf.matmul(queries, model.char_embeddings, transpose_b=True)
            logits = logits + self.bert_bias
            self.raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits
            )

        self.loss = tf.losses.compute_weighted_loss(tf.reduce_mean(self.raw_loss), self.weight)


class Tagger(object):
    def __init__(self, index: int, cfg: Config, model: 'MorphModel'):
        self.name = cfg.get_string('name')
        self.index = index
        self.debug = cfg.get_bool('debug', model.debug)
        self.cfg = cfg
        self.model = model
        with tf.variable_scope(self.name) as sc:
            self.scope = sc
            self.tags = model.dicts[self.name]

            self.model_output, self.len_handler = model.prediction(self.name)  # b x t x d

            self.embed_size = self.model_output.shape[-1].value
            assert self.embed_size is not None

            self.tag_embeddings = tf.get_variable(
                'embeds',
                shape=[self.tags.size, self.embed_size],
                dtype=model.dtype,
                initializer=tf.random_normal_initializer(
                    dtype=model.dtype,
                    stddev=1 / np.sqrt(self.embed_size)
                )
            )

            self.tag_data = model.data.tags[:, index]

            shaped_data = tf.reshape(self.model_output,
                                     [model.data.batch_size * self.len_handler.batch_mlen, self.embed_size])
            shaped_logits = tf.matmul(shaped_data, self.tag_embeddings, transpose_b=True)
            self.logits = tf.reshape(shaped_logits,
                                     [model.data.batch_size, self.len_handler.batch_mlen, self.tags.size])
            self.normalizer = tf.reduce_logsumexp(self.logits, axis=2)

            smooth = cfg.get_float('smoothing', 0.0)
            exp_smooth = cfg.get_bool('exp_smoothing', False)

            self.wide_logits = self.len_handler.scatter(self.logits)

            use_crf = cfg.get_bool('crf', False)

            if use_crf:
                self.crf_matrix = tf.get_variable(
                    name='crf_transitions',
                    shape=[self.tags.size, self.tags.size],
                    dtype=self.logits.dtype,
                    initializer=tf.random_normal_initializer(
                        dtype=self.logits.dtype
                    )
                )

                zero_tags = tf.equal(self.tag_data, 0)
                zero_tags = tf.expand_dims(zero_tags, -1)
                one_hot_tags = tf.one_hot(self.tag_data, self.tags.size, dtype=tf.bool, on_value=True, off_value=False)
                crf_tag_bitmap = tf.logical_or(zero_tags, one_hot_tags)

                crf_seq_score = tf.contrib.crf.crf_multitag_sequence_score(
                    inputs=self.logits,
                    tag_bitmap=crf_tag_bitmap,
                    sequence_lengths=model.data.length,
                    transition_params=self.crf_matrix
                )

                crf_normalizer = tf.contrib.crf.crf_log_norm(
                    inputs=self.logits,
                    sequence_lengths=model.data.length,
                    transition_params=self.crf_matrix
                )

                self.crf_loss = crf_normalizer - crf_seq_score
            elif smooth == 0.0:
                self.raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.tag_data,
                    logits=self.wide_logits
                )
            elif exp_smooth:
                self.raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.tag_data,
                    logits=self.wide_logits
                )
                correct = 1.0 - smooth
                if cfg.get_bool('one_sided', False):
                    self.raw_loss = tf.maximum(self.raw_loss + tf.log(correct), 0)
                else:
                    self.raw_loss = tf.square(self.raw_loss + tf.log(correct))
            else:
                correct = 1.0 - smooth
                incorrect = smooth / (self.tags.size - 1)
                to_guess = tf.one_hot(
                    indices=self.tag_data,
                    depth=self.tags.size,
                    on_value=correct,
                    off_value=incorrect,
                    dtype=self.logits.dtype
                )
                self.raw_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=to_guess,
                    logits=self.wide_logits
                )

            if self.debug:
                batch_ids = tf.range(model.data.batch_size)  # b
                batch_ids = tf.tile(tf.expand_dims(batch_ids, axis=1), [1, model.data.batch_mlen])  # b x l
                len_ids = tf.range(model.data.batch_mlen)  # l
                len_ids = tf.tile(tf.expand_dims(len_ids, axis=0), [model.data.batch_size, 1])
                indices = tf.stack([batch_ids, len_ids, self.tag_data], axis=2)

                gold_logits = tf.gather_nd(
                    params=self.wide_logits,  # b x l x k
                    indices=indices  # b x l x 3
                )
                tf.summary.histogram(f"normalizer", self.normalizer, family='tagprob')
                tf.summary.histogram(f"correct", gold_logits, family='tagprob')
                tf.summary.histogram(f"prob", tf.exp(gold_logits - self.normalizer), family='tagprob')

            self.mask = tf.not_equal(self.tag_data, 0)
            self.raw_weight = self.weight = self.cfg.get_float('weight', 1.0)

            if use_crf:
                self._init_crf()
            else:
                self._init_argmax()

            if self.raw_weight == 0:
                self.avg_loss = tf.constant(0, dtype=tf.float32, shape=[])

    def _init_crf(self):
        self.raw_answers, self.crf_scores = tf.contrib.crf.crf_decode(
            potentials=self.logits,
            transition_params=self.crf_matrix,
            sequence_length=self.model.data.length
        )
        self.answers = self.raw_answers
        self.hits = tf.equal(self.answers, self.tag_data) & self.mask
        self.avg_loss = self.weight * tf.reduce_mean(self.crf_loss)

    def _init_argmax(self):
        self.raw_answers = tf.argmax(self.logits, axis=-1, output_type=self.tag_data.dtype)
        self.answers = self.len_handler.scatter(self.raw_answers)

        self.hits = tf.equal(self.answers, self.tag_data) & self.mask

        waf = self.cfg.get_config('weight_acc_filter', None)
        if waf is not None:
            tag = waf.get_string('tag')
            border = waf.get_float('border')
            acc = self.model.accuracy(tag)
            exp_sm = waf.get_float('exp', None)
            if exp_sm is not None:
                diff = tf.minimum(acc - border, 0.0)
                diff = diff * exp_sm
                self.weight = tf.exp(diff) * self.weight
            else:
                cond = acc > border
                self.weight = tf.where(cond, self.weight, 0.0)
            if self.debug:
                tf.summary.scalar(f"weight", self.weight, family="debug")
                tf.summary.scalar(f"tacc", acc, family="debug")
            if not self.model.infer:
                print("gating", self.name, "tag loss on", tag, border, file=sys.stderr)

        self.weight = tf.to_float(self.mask) * self.weight
        self.avg_loss = tf.losses.compute_weighted_loss(self.raw_loss, self.weight)


class MorphModel(object):
    def __init__(self, cfg: Config, data: InputData, dics: Dictionaries, infer: bool, bert_cfg):
        self.layers = []
        self.taggers: typing.Dict[str, Tagger] = {}
        self.cfg = cfg
        self.data = data
        self.debug = cfg.get_bool('debug', False)
        self.dicts = dics
        self.dtype = tf.float32
        self.bos_mask = tf.constant(True, dtype=tf.bool, shape=[1, 1])
        self.max_length = cfg.get_int('max_length', 200)
        self.num_taggers = len(cfg.get_list('tags'))
        self.infer = infer
        self.scope = tf.get_variable_scope()
        self.bert_cfg = bert_cfg
        self.bert: typing.Optional[BertObjective] = None

        self.char_emb_size = cfg.get_int('char.embed.size', 16)

        self.char_dict = dics['chars']

        self.char_embeddings = tf.get_variable(
            'char_embeds',
            shape=[self.char_dict.size, self.char_emb_size],
            dtype=self.dtype,
            initializer=tf.random_normal_initializer(
                dtype=self.dtype,
                stddev=1 / np.sqrt(self.char_emb_size)
            )
        )

        self.embedded_chars = tf.nn.embedding_lookup(
            params=self.char_embeddings,
            ids=data.chars
        )

        self.global_step: tf.Variable = tf.train.get_or_create_global_step()

        self.num_sentences: tf.Variable = tf.get_variable(
            name='num_sentences',
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False
        )

        self.inc_num_sentences = self.num_sentences.assign_add(tf.to_int64(data.batch_size))
        self.tagger_inputs = {}
        self.num_layers = -1
        self.tagger_info = {}
        self.len_handler = data
        self.acc_cache = {}

    def prediction(self, name):
        return self.tagger_inputs[name]

    def initialize(self, layer_cfgs, tagger_cfgs):
        self.count_layers(layer_cfgs)
        self.compute_tagger_info(tagger_cfgs)

        idx = 0
        state = self.embedded_chars
        for cfg in layer_cfgs:
            repeat = cfg.get_int('repeat', 1)
            ctor = Layers.for_name(cfg.type)
            for _ in range(repeat):
                init_state = state
                with tf.variable_scope(f'l_{idx}_{cfg.type}'):
                    impl = ctor(
                        cfg=cfg,
                        model=self,
                        input=state,
                        name=f'l_{idx}_{cfg.type}'
                    )
                    self.layers.append(impl)
                    state = impl.output

                    if self.len_handler is not impl.length_handler:
                        self.len_handler = impl.length_handler

                    if cfg.get_bool('normalize', False):
                        state = trf.layer_norm(state)

                    dropout_keep = cfg.get_float('dropout_keep', 1.0)
                    if dropout_keep != 1.0 and not self.infer:
                        state = tf.nn.dropout(state, dropout_keep)

                    if cfg.get_bool('concat', False) and not impl.concat_handled:
                        state = tf.concat([init_state, state], axis=2)

                    self.build_taggers_at(idx, impl)

                    idx += 1

        if not self.infer and self.bert_cfg is not None:
            self.bert = BertObjective(self.data, self.cfg, self)

    def count_layers(self, layer_cfgs):
        idx = 0
        for cfg in layer_cfgs:
            idx += cfg.get_int('repeat', 1)
        self.num_layers = idx

    def compute_tagger_info(self, tagger_defs):
        default_tag_size = self.cfg.get_int('embed.tags')

        last_layer = self.num_layers - 1
        idx = 0
        for t in tagger_defs:
            layer = t.get_int('source', last_layer)
            name = t.get_string('name')
            by_size = self.tagger_info.setdefault(layer, [])
            size = t.get_int('embed', default_tag_size)
            by_size.append((name, size, t, idx))
            idx = t.get_int('index', idx + 1)

    def build_taggers_at(self, idx, layer):
        info = self.tagger_info.get(idx, [])
        if not info:
            return

        layer_sizes = [x[1] for x in info]
        total_size = sum(layer_sizes)
        input = layer.output
        if idx != (self.num_layers - 1) and not self.infer:
            tag_names = [x[0] for x in info]
            print("tags", tag_names, "are connected to", input.name, file=sys.stderr)

        projected = tf.layers.dense(input, total_size, activation=tf.nn.selu, name=f"l_{idx}")
        if len(layer_sizes) == 1:
            self.tagger_inputs[info[0][0]] = projected, layer.length_handler
        else:
            split = tf.split(projected, layer_sizes, axis=2, name=f'split_l_{idx}')
            for x, tens in zip(info, split):
                self.tagger_inputs[x[0]] = tens, layer.length_handler

        for name, size, cfg, idx in info:
            tagger = Tagger(
                index=idx,
                cfg=cfg,
                model=self
            )
            self.taggers[tagger.name] = tagger

    def tag_lookups(self, *names):
        tensors = []
        dics = []
        for n in names:
            tag = self.taggers[n]
            tensors.append(tag.answers)
            dics.append(tag.tags)
        tensors = tf.stack(tensors, axis=2)
        return tensors, dics

    def tag_probs(self, *names):
        tensors = []
        for n in names:
            tag = self.taggers[n]
            probs = tf.reduce_max(tag.logits, axis=-1) - tag.normalizer
            tensors.append(probs)
        val = tf.stack(tensors, axis=2)
        return tf.exp(val)

    def accuracy(self, tag_name):
        cached = self.acc_cache.get(tag_name, None)
        if cached is not None:
            return cached

        tag = self.taggers[tag_name]
        hits = tf.reduce_sum(tf.to_float(tag.hits))
        total = tag.mask
        total = tf.to_float(total)
        total = tf.reduce_sum(total)
        acc = hits / tf.maximum(total, 1.0)
        self.acc_cache[tag_name] = acc
        return acc
