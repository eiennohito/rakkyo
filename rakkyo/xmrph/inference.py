import typing

import tensorflow as tf
import numpy as np
import sys
from .infer_base import BasicInferrer
import csv


class JumanInferrer(BasicInferrer):
    def __init__(self, ma):
        super().__init__(ma)

    def format_result(self, writer, comment, data, raw_tags):
        lookups = self.tag_lookups
        if len(comment) > 0:
            writer.write(comment.decode('utf-8'))
            writer.write('\n')

        tags = raw_tags[1:]

        start = 0

        for end in range(1, len(data)):
            seg_tag = tags[end, 0]
            if seg_tag == 1:  # B
                fields = [
                    data[start:end], "*", "*",
                    lookups[1].tostr(tags[start, 1]), "0",
                    lookups[2].tostr(tags[start, 2]), "0",
                    lookups[3].tostr(tags[start, 3]), "0",
                    lookups[4].tostr(tags[start, 4]), "0",
                    "NIL"
                ]
                writer.write(" ".join(fields))
                writer.write("\n")
                start = end
        fields = [
            data[start:], "*", "*",
            lookups[1].tostr(tags[start, 1]), "0",
            lookups[2].tostr(tags[start, 2]), "0",
            lookups[3].tostr(tags[start, 3]), "0",
            lookups[4].tostr(tags[start, 4]), "0",
            "NIL"
        ]
        writer.write(" ".join(fields))
        writer.write("\nEOS\n")


class MrphInferrer(BasicInferrer):
    def __init__(self, ma):
        super().__init__(ma)

    def format_result(self, writer, comment, data, raw_tags):
        lookups = self.tag_lookups
        tags = raw_tags[1:]

        start = 0

        for end in range(1, len(data)):
            seg_tag = tags[end, 0]
            if seg_tag == 1:  # B
                surf = data[start:end]
                pos = lookups[1].tostr(tags[start, 1])
                spos = lookups[2].tostr(tags[start, 2])
                writer.write(f'{surf}_{pos}:{spos} ')
                start = end

        surf = data[start:]
        pos = lookups[1].tostr(tags[start, 1])
        spos = lookups[2].tostr(tags[start, 2])
        writer.write(f'{surf}_{pos}:{spos}')

        if len(comment) > 0:
            writer.write(' ')
            writer.write(comment.decode('utf-8'))
            writer.write('\n')


class DebugInferrer(BasicInferrer):
    def __init__(self, ma):
        super().__init__(ma)
        self.fetches['probs'] = ma.model.tag_probs(
            'seg', 'pos', 'subpos', 'ctype', 'cform'
        )

    def format_result(self, writer, comment, data, raw_tags):
        lookups = self.tag_lookups
        if len(comment) > 0:
            writer.write(comment.decode('utf-8'))
            writer.write('\n')

        tags = raw_tags[1:]
        probs = self.ctx['probs'][self.idx, 1:] * 100

        for idx in range(len(data)):
            fields = [
                data[idx],
                lookups[0].tostr(tags[idx, 0]),
                lookups[1].tostr(tags[idx, 1]),
                lookups[2].tostr(tags[idx, 2]),
                lookups[3].tostr(tags[idx, 3]),
                lookups[4].tostr(tags[idx, 4]),
                '{0:.1f}'.format(probs[idx, 0]),
                '{0:.1f}'.format(probs[idx, 1]),
                '{0:.1f}'.format(probs[idx, 2]),
                '{0:.1f}'.format(probs[idx, 3]),
                '{0:.1f}'.format(probs[idx, 4]),
            ]
            writer.write("\t".join(fields))
            writer.write("\n")
        writer.write("EOS\n")


class Inference(object):
    def __init__(self, ma):
        self.ma = ma
        self.chars = self._read_chars(ma.model.cfg)
        mode = ma.model.cfg.get_string('infer_fmt', 'juman')
        if mode == 'juman':
            self.runner = JumanInferrer(ma)
        elif mode == 'mrph':
            self.runner = MrphInferrer(ma)
        elif mode == 'debug':
            self.runner = DebugInferrer(ma)
        elif mode == 'attnviz':
            from .attn_vis_infer import AttentionViz
            self.runner = AttentionViz(ma)
        else:
            raise NotImplementedError("Invalid mode:", mode)

    @staticmethod
    def codepts(file, chars):
        def impl():
            comment = ""
            with open(file, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.rstrip('\n')
                    if line.startswith('#'):
                        comment = line
                    else:
                        cpts = [2]
                        for c in line:
                            cpts.append(chars.get(c, 1))
                        cpts.append(3)
                        cpts = np.array(cpts, dtype=np.int32)
                        yield {
                            'comment': comment,
                            'raw': line,
                            'chars': cpts
                        }

        return impl

    def enhance(self, x):
        empty_tags = tf.constant(0, shape=[self.ma.num_tags, 0], dtype=tf.int32)
        x['length'] = tf.shape(x['chars'])[0]
        x['tags'] = empty_tags
        return x

    def enhance2(self, x):
        x['chars_orig'] = x['chars']
        return x

    def run(self, names):
        with self.ma.graph.as_default():
            self._run(names)

    def _run(self, names):
        d = tf.data.Dataset.from_generator(
            Inference.codepts(names, self.chars),
            output_types={
                'comment': tf.string,
                'raw': tf.string,
                'chars': tf.int32
            },
            output_shapes={
                'comment': [],
                'raw': [],
                'chars': [None]
            }
        )

        d = d.map(self.enhance)

        d = d.padded_batch(
            batch_size=self.ma.model.cfg.get_int('infer_batch', 5),
            padded_shapes={
                'comment': [],
                'raw': [],
                'tags': [self.ma.num_tags, 0],
                'chars': [None],
                'length': []
            }
        )

        d = d.map(self.enhance2)

        iterator = d.make_one_shot_iterator()
        feed_dict = self.ma.model.data.feed_dict(self.ma.sess, iterator)

        try:
            while True:
                self.runner.run_inference(feed_dict, sys.stdout)
        except tf.errors.OutOfRangeError:
            pass

    def _read_chars(self, cfg):
        fname: str = cfg.get_string('dics.chars')
        fname = fname.replace('.vdic', '.dic')
        return read_chardic(fname)


def read_chardic(fname) -> typing.Dict[str, int]:
    result = {}
    with open(fname, 'rt', encoding='utf-8', newline='') as fl:
        idx = 4
        for elems in csv.reader(fl, delimiter='\t', quoting=csv.QUOTE_NONE):
            result[elems[0]] = idx
            idx += 1
    return result
