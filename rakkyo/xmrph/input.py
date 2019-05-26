import tensorflow as tf
import sys
from pyhocon import ConfigTree as Config


class InputPipeline(object):
    def parse_example(self, ex: tf.Tensor, num_tags: int):
        with tf.device('/cpu:0'):
            feature_map = {
                'chars': tf.VarLenFeature(tf.int64),
                'tags': tf.VarLenFeature(tf.int64)
            }

            if self.local_bert and self.global_bert:
                feature_map['bert'] = tf.VarLenFeature(tf.int64)

            ex = tf.parse_single_example(
                serialized=ex,
                features=feature_map
            )

            chars = tf.to_int32(ex['chars'].values)
            char_len = tf.shape(chars)[0]
            tags = tf.to_int32(tf.reshape(ex['tags'].values, [char_len, num_tags]))
            tags = tf.transpose(tags, [1, 0])

            result = dict(chars=chars, length=char_len, tags=tags)

            if self.global_bert:
                if self.local_bert:
                    result['bert'] = tf.cast(ex['bert'].values, tf.int32)
                else:
                    result['bert'] = tf.zeros_like(chars, dtype=tf.int32)

            return result

    def make_boundaries(self, start, end, increase):
        value = int(start)
        result = [value]
        while value < end:
            value = int(value * increase)
            result.append(value)
        return result

    def make_batch_sizes(self, num_data, boundaries):
        result = []
        for x in boundaries:
            result.append(int(num_data / x) + 1)
        result.append(1)
        return result

    def _read_file(self, nm):
        return tf.data.TFRecordDataset(
            nm,
            compression_type=self.cfg.get_string('compression', ''),
            buffer_size=self.cfg.get_int('input_buffer', 4 * 1024 * 1024)
        )

    def _read_source(self, cfg):
        name_pat = cfg.get_string('pattern')
        print("train pattern: ", name_pat, file=sys.stderr)
        files = tf.data.Dataset.list_files(
            name_pat,
            shuffle=False
        )

        d = files.repeat(
            count=cfg.get_int('epochs', 1)
        )

        shuffle_files = cfg.get_int('shuffle_files', None)
        if shuffle_files is not None:
            d = d.shuffle(shuffle_files)

        pread = cfg.get_int('parallel_reads', 4)
        if pread > 1:
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    self._read_file,
                    cycle_length=pread,
                    sloppy=True
                )
            )
        else:
            d = d.flat_map(
                self._read_file
            )
        return d

    def _read_sources(self, src_cfgs):
        indices = []
        datasets = []
        for idx, cfg in enumerate(src_cfgs):
            src = self._read_source(cfg)
            datasets.append(src)
            wnd = cfg.get_int('window', 1)
            print("with window", wnd, file=sys.stderr)
            for _ in range(wnd):
                indices.append(idx)
        idx_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(indices, dtype=tf.int64)).repeat()
        chosen = tf.contrib.data.choose_from_datasets(datasets, idx_dataset)
        return chosen

    def preprocess_bert(self, tensors):
        bert = tensors['bert']
        chars_orig = chars = tensors['chars']

        mask_id = self.cfg.get_int('bert_mask_id')
        char_file = self.cfg.get_string('bert_char_file')
        char_max = self.cfg.get_int('bert_char_max')
        char_bias = self.cfg.get_int('bert_char_bias', 100)

        char_table = []
        char_freqs = []

        from .inference import read_chardic

        raw_chardic = read_chardic(self.cfg.get_string('bert_char_dic'))

        line_no = 0
        with open(char_file, 'rt') as fl:
            for line in fl:
                parts = line.split('\t', 2)
                id = raw_chardic.get(parts[0])
                if id is not None:
                    char_table.append(id)
                    char_freqs.append(int(parts[1]))

                line_no += 1
                if line_no > char_max:
                    break

        char_freqs = tf.log(tf.sqrt(tf.cast(char_freqs, tf.float32) + char_bias))
        char_freqs = tf.reshape(char_freqs, [1, -1])
        char_table = tf.cast(char_table, tf.int32)

        char_shape = tf.shape(chars)
        num_samples = tf.reduce_prod(char_shape)
        random_cids = tf.multinomial(char_freqs, num_samples, output_dtype=tf.int32)
        random_cids = tf.reshape(random_cids, char_shape)
        random_chars = tf.nn.embedding_lookup(char_table, random_cids)
        masked_ids = tf.ones_like(chars, dtype=tf.int32) * mask_id

        rand_pos = tf.equal(bert, 2)
        mask_pos = tf.equal(bert, 3)
        # chars = tf.Print(chars, tf.shape_n([chars, bert, masked_ids]), first_n=10)
        chars = tf.where(mask_pos, masked_ids, chars)
        chars = tf.where(rand_pos, random_chars, chars)

        tensors['chars'] = chars
        return tensors

    def add_chars_orig(self, tens):
        tens['chars_orig'] = tens['chars']
        return tens

    def __init__(self, cfg: Config, num_tags: int, global_bert: bool):
        self.global_bert = global_bert
        self.local_bert = cfg.get_bool('bert', False)
        self.cfg = cfg
        self.bucket_boundaries = self.make_boundaries(
            start=cfg.get_int('bucket_min', 8),
            end=cfg.get_int('bucket_max', 200),
            increase=cfg.get_float('bucket_step', 1.2)
        )

        srcs = cfg.get_list('sources', [])
        if len(srcs) == 0:
            d = self._read_source(cfg)
        else:
            d = self._read_sources(srcs)

        shuffle = cfg.get_int('shuffle_buffer', 10000)
        if shuffle > 0:
            d = d.shuffle(
                buffer_size=shuffle,
                seed=cfg.get_int('seed', 0xdeadbeef)
            )

        d = d.map(
            map_func=lambda x: self.parse_example(x, num_tags),
            num_parallel_calls=cfg.get_int('parallel_parse', 4)
        )

        padded_shapes = dict(chars=[None], length=[], tags=[num_tags, None])
        if self.global_bert:
            padded_shapes['bert'] = [None]

        d = d.apply(
            tf.contrib.data.bucket_by_sequence_length(
                lambda x: x['length'],
                bucket_boundaries=self.bucket_boundaries,
                bucket_batch_sizes=self.make_batch_sizes(cfg.get_int('batch', 160), self.bucket_boundaries),
                padded_shapes=padded_shapes
            )
        )

        d = d.map(self.add_chars_orig)

        if self.local_bert and self.global_bert:
            d = d.map(self.preprocess_bert)

        d = d.prefetch(cfg.get_int('prefetch', 4))

        if cfg.get_bool('prefetch_gpu', False):
            d = d.apply(
                tf.contrib.data.prefetch_to_device(
                    device="/gpu:0"
                )
            )

        self.pipeline = d


class InputData(object):
    def __init__(self, num_tags: int, dataset: tf.data.Dataset, infer: bool, bert: bool):
        self.infer = infer
        self.num_tags = num_tags
        self.handle = tf.placeholder(tf.string, shape=[])
        otype = dataset.output_types
        oshp = dataset.output_shapes
        if infer:
            otype['raw'] = tf.string
            otype['comment'] = tf.string
            oshp['raw'] = [None]
            oshp['comment'] = [None]
        if bert:
            otype['bert'] = tf.int32
            oshp['bert'] = [None, None]

        self.out_types = otype
        self.out_shapes = oshp

        self.iterator: tf.data.Iterator = tf.data.Iterator.from_string_handle(
            string_handle=self.handle,
            output_types=otype,
            output_shapes=oshp
        )

        self.next = self.iterator.get_next()
        self.chars: tf.Tensor = self.next['chars']
        self.tags: tf.Tensor = self.next['tags']
        self.length: tf.Tensor = self.next['length']
        self.chars_shape: tf.Tensor = tf.shape(self.chars)
        self.batch_size: tf.Tensor = self.chars_shape[0]
        self.batch_mlen: tf.Tensor = self.chars_shape[1]

        self.len_bool_mask = tf.sequence_mask(self.length)
        over_len = tf.logical_not(self.len_bool_mask)
        self.length_softmax_mask = -1e9 * tf.to_float(over_len)

    def handle_for(self, sess: tf.Session, actual: tf.data.Iterator):
        return sess.run(actual.string_handle())

    def feed_dict(self, sess, actual: tf.data.Iterator):
        return {self.handle: self.handle_for(sess, actual)}

    def scatter(self, x):
        return x


class TagDictionary(object):
    @staticmethod
    def load_from(cfg):
        with open(cfg, encoding='utf-8') as f:
            lines = [x[:-1] for x in f if len(x) > 1]
            return TagDictionary(lines, cfg)

    def __init__(self, lines, path):
        self.data = lines
        self.path = path
        self.ids = dict((v, i) for i, v in enumerate(self.data))

    def init(self):
        return tf.no_op()

    @property
    def size(self):
        return len(self.data)

    def tostr(self, v):
        return self.data[v]

    def toid(self, x):
        return self.ids[x]


class Dictionaries(object):
    def __init__(self, cfg: Config):
        self.data = dict()

        for k, v in cfg.items():
            loaded = TagDictionary.load_from(v)
            self.data[k] = loaded

    def init_lookups(self, sess: tf.Session):
        inits = [v.init() for v in self.data.values()]
        sess.run(inits)

    def __getitem__(self, item):
        return self.data[item]
