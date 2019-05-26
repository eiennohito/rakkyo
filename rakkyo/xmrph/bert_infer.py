from .model import MorphModel
from .input import InputData
from .inference import read_chardic
import tensorflow as tf
import numpy as np


class BertInferenceInput(object):
    def __init__(self, data: InputData, chars, fname):
        self.data = data
        self.chars = chars
        self.line = None
        self.fname = fname

    def readlines(self):
        empty_int32 = np.asarray([], dtype=np.int32)
        empty_tags = np.reshape(empty_int32, [5, 0])
        for line in open(self.fname):
            self.line = line
            codes = [self.chars.get(c, 2) for c in line.rstrip('\n')]
            yield {
                'chars': np.asarray(codes, dtype=np.int32),
                'tags': empty_tags,
                'bert': empty_int32
            }

    def iterator(self):
        dset = tf.data.Dataset.from_generator(
            self.readlines,
            self.data.out_types,
            self.data.out_shapes
        )

        dset = dset.map(lambda x: dict((k, tf.expand_dims(v, 0)) for k, v in x))

        return dset.make_one_shot_iterator()


class BertInference(object):
    def __init__(self, model: MorphModel):
        self.model = model
        self.chardic_file = model.cfg.get('input.train.bert_char_dic')
        self.chardic = read_chardic(self.chardic_file)
        fname = model.cfg.get('infer')
        self.input = BertInferenceInput(model.data, self.chardic, fname)

    def run(self, sess: tf.Session):
        iter = self.input.iterator()
        handle = sess.run(iter.string_handle())

        states = self.model.bert.queries
        states = tf.reshape(states, [self.model.data.batch_mlen, -1])

        logits = tf.matmul(states, self.model.char_embeddings, transpose_b=True)
        logits = logits + self.model.bert.bert_bias

        values, indices = tf.nn.top_k(logits, k=10)
        cdict = self.model.char_dict

        while True:
            v, i = sess.run([values, indices], feed_dict={self.model.data.handle: handle})
            line = self.input.line
            for idx, c in enumerate(line):
                scores = []
                for vx, ix in zip(v[idx], i[idx]):
                    ch = cdict.tostr(ix)
                    scores.append(f'{ch}:{vx:.2f}')
                print(f'{c}\t{",".join(scores)}')
            print("EOS")
