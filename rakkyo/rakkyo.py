import tensorflow as tf
from input.config import parse_config_args
import numpy as np
from pyhocon import ConfigTree as Config
import os
import re
import time
from model.metrics import Metrics, Windowed, RateCalculator
import sys

from xmrph.model import MorphModel
from xmrph.input import InputPipeline, InputData, Dictionaries


class Optimizer(object):
    def __init__(self, cfg: Config, model: MorphModel):
        subcfg = cfg.get_config("optimizer", Config())

        warmup = subcfg.get_int('warmup', 1000)
        decay = subcfg.get_bool('decay', True)

        step_flt = tf.to_float(model.global_step)
        base_rate = subcfg.get_float('lrate', model.char_emb_size ** -0.5)

        if decay:
            if subcfg.get_bool('no_warmup', False):
                warmup_rate = warmup ** -0.5
            else:
                warmup_rate = step_flt * warmup ** -1.5
            self.lrate = base_rate * tf.minimum(tf.rsqrt(step_flt), warmup_rate)
        else:
            self.lrate = tf.minimum(step_flt / warmup, 1.0) * base_rate

        opt = subcfg.get_string('type', 'adam')
        if opt == 'adam':
            self.opt = tf.train.AdamOptimizer(
                learning_rate=self.lrate,
                beta1=subcfg.get_float('beta1', 0.9),
                beta2=subcfg.get_float('beta2', 0.98),
                epsilon=subcfg.get_float('eps', 1e-9)
            )
        elif opt == 'ladam':
            self.opt = tf.contrib.opt.LazyAdamOptimizer(
                learning_rate=self.lrate,
                beta1=subcfg.get_float('beta1', 0.9),
                beta2=subcfg.get_float('beta2', 0.98),
                epsilon=subcfg.get_float('eps', 1e-9)
            )
        elif opt == 'ggt':
            self.opt = tf.contrib.opt.GGTOptimizer(
                learning_rate=self.lrate,
                beta1=subcfg.get_float('beta1', 0.9),
                window=subcfg.get_int('window', 10),
                eps=subcfg.get_float('eps', 1e-4)
            )

        self.loss = tf.losses.get_total_loss()
        self.grads_and_vars = self.opt.compute_gradients(self.loss)

        self.opt_op = self.opt.apply_gradients(self.grads_and_vars, model.global_step)

        draws = subcfg.get_list('draw_grads', [])
        if len(draws) > 0:
            draws = [re.compile(x) for x in draws]

            for g, v in self.grads_and_vars:
                for rx in draws:
                    if rx.match(v.name) is not None:
                        tf.summary.histogram('grad/' + v.name, g)
                        break


class TrainSummaries(object):
    def __init__(self, ma):
        mdl: MorphModel = ma.model

        tf.summary.scalar('misc/lrate', ma.optimizer.lrate)
        tf.summary.scalar('misc/step', ma.model.global_step)
        if tf.test.is_gpu_available():
            tf.summary.scalar('misc/gpu_max_mem', tf.contrib.memory_stats.MaxBytesInUse())
        tag_non_zero = tf.not_equal(mdl.data.tags, 0)
        tag_non_zero = tag_non_zero & tf.expand_dims(mdl.data.len_bool_mask, 1)
        num_non_zero = tf.reduce_sum(tf.to_int32(tag_non_zero))
        total_num = tf.reduce_sum(mdl.data.length) - tf.shape(mdl.data.length)[0]  # BOS tags are always zero
        non_zero_ratio = tf.to_float(num_non_zero) / tf.maximum(tf.to_float(total_num) * ma.num_tags, 1.0)
        tf.summary.scalar('misc/non_zero', non_zero_ratio)
        tf.summary.scalar('misc/bsize', mdl.data.batch_size)
        tf.summary.scalar('misc/bmlen', mdl.data.batch_mlen)

        self.rate = RateCalculator(mdl.data.batch_size, 20, mdl.global_step, 'example_rate')
        tf.summary.scalar('misc/rate', self.rate.rate(gated=True))

        tf.summary.scalar('loss/all', ma.optimizer.loss)
        for t in mdl.taggers.values():
            tf.summary.scalar(f'loss/{t.name}', t.avg_loss)

        seg_name = mdl.cfg.get_string('seg_name', 'seg')
        seg = mdl.taggers[seg_name]

        seg_b = tf.equal(seg.tag_data, 1)
        seg_tp = tf.equal(seg.answers, 1) & seg_b
        seg_fn = tf.not_equal(seg.answers, 1) & seg_b
        seg_fp = tf.equal(seg.answers, 1) & tf.not_equal(seg.tag_data, 1) & seg.mask

        seg_tp = tf.maximum(tf.reduce_sum(tf.to_int32(seg_tp)) - mdl.data.batch_size, 0)
        seg_fn = tf.maximum(1, tf.reduce_sum(tf.to_int32(seg_fn)))
        seg_fp = tf.maximum(1, tf.reduce_sum(tf.to_int32(seg_fp)))
        seg_tp = tf.to_float(seg_tp)
        seg_fp = tf.to_float(seg_fp)
        seg_fn = tf.to_float(seg_fn)
        seg_prec = seg_tp / (seg_tp + seg_fp)
        seg_rec = seg_tp / (seg_tp + seg_fn)

        self.moving_upd = []
        self.moving_upd.extend(self.rate.updates)
        step = mdl.global_step
        window_size = mdl.cfg.get_int('summary.train.window', 20)
        seg_prec_wnd = Windowed(seg_prec, window_size, step, "train/summ/wnd/prec")
        seg_rec_wnd = Windowed(seg_rec, window_size, step, "train/summ/wnd/rec")
        self.moving_upd.append(seg_prec_wnd.update)
        self.moving_upd.append(seg_rec_wnd.update)
        seg_prec = seg_prec_wnd.mean_updated
        seg_rec = seg_rec_wnd.mean_updated

        seg_f1 = 2 * seg_prec * seg_rec / tf.maximum(seg_prec + seg_rec, 1e-6)

        tf.summary.scalar('train/seg/precision', seg_prec)
        tf.summary.scalar('train/seg/recall', seg_rec)
        tf.summary.scalar('train/seg/f1', seg_f1)

        from tensorboard.plugins import projector
        conf = projector.ProjectorConfig()

        char_embed = conf.embeddings.add()
        char_embed.tensor_name = ma.model.char_embeddings.name
        char_embed.metadata_path = ma.model.char_dict.path

        for tag in mdl.taggers.values():
            hits = tf.reduce_sum(tf.to_float(tag.hits))
            total = tf.maximum(tf.reduce_sum(tf.to_float(tag.mask)), 1.0)
            accuracy = hits / total
            acc_wnd = Windowed(accuracy, window_size, step, f"train/summ/wnd/acc/{tag.name}")
            self.moving_upd.append(acc_wnd.update)
            tf.summary.scalar(f'train/{tag.name}/acc', acc_wnd.mean_updated)

            tag_embed = conf.embeddings.add()
            tag_embed.tensor_name = tag.tag_embeddings.name
            tag_embed.metadata_path = tag.tags.path

        projector.visualize_embeddings(ma.summary_writer, conf)
        self.update_avg = tf.group(self.moving_upd)

        if mdl.bert is not None:
            tf.summary.scalar('loss/+bert', mdl.bert.loss)


class TimedExecution(object):
    def __init__(self, period, callback, *fetches):
        self.last_run = time.monotonic()
        self.period = period
        self.callback = callback
        self.fetches = fetches

    def should_run(self, now):
        return (now - self.last_run) > self.period


class Evaluator(object):
    def __init__(self, cfg: Config, ma, name, mdl: MorphModel):
        self.name = name
        self.pipeline = InputPipeline(cfg, ma.num_tags, mdl.bert_cfg is not None)
        self.iterator = self.pipeline.pipeline.make_initializable_iterator()
        self.metrics = Metrics(name)
        self.debug = cfg.get_bool('debug', False)

        self.mdl = mdl
        self.ma = ma

        seg_name = mdl.cfg.get_string('seg_name', 'seg')
        seg = mdl.taggers[seg_name]

        seg_ans_b = tf.equal(seg.answers, 1) & seg.mask
        seg_dat_b = tf.equal(seg.tag_data, 1)
        seg_hits = seg_ans_b & seg_dat_b
        seg_ans_cum = tf.cumsum(tf.to_int32(seg_ans_b), axis=-1)
        seg_dat_cum = tf.cumsum(tf.to_int32(seg_dat_b), axis=-1)
        self.seg_hit_idxs = tf.where(seg_hits)
        seg_ans_g = tf.gather_nd(seg_ans_cum, self.seg_hit_idxs)
        seg_dat_g = tf.gather_nd(seg_dat_cum, self.seg_hit_idxs)
        seg_hits_cum = tf.cumsum(tf.to_int32(seg_hits), axis=-1, exclusive=True)
        idx2 = tf.gather_nd(seg_hits_cum, self.seg_hit_idxs)
        compr_idxs = tf.stack([self.seg_hit_idxs[:, 0], tf.to_int64(idx2)], axis=1)
        compr_shape = tf.to_int64(mdl.data.chars_shape)
        seg_ans_den = tf.sparse_to_dense(compr_idxs, compr_shape, seg_ans_g, default_value=0)
        seg_dat_den = tf.sparse_to_dense(compr_idxs, compr_shape, seg_dat_g, default_value=0)
        seg_ans_diff = seg_ans_den[:, 1:] - seg_ans_den[:, :-1]
        seg_dat_diff = seg_dat_den[:, 1:] - seg_dat_den[:, :-1]

        seg_tok_tp = tf.equal(seg_ans_diff, 1) & tf.equal(seg_dat_diff, 1)
        self.seg_tok_metric = self.metrics.eval_measures_den(seg_tok_tp, seg_ans_b, seg_dat_b,
                                                             scope="seg_tok", penalty=-mdl.data.batch_size)

        seg_bnd_tp = seg_ans_b & seg_dat_b
        seg_bnd_fn = tf.not_equal(seg.answers, 1) & seg_dat_b
        seg_bnd_fp = seg_ans_b & tf.not_equal(seg.tag_data, 1)

        # subtract first B's
        self.seg_bnd_metric = self.metrics.eval_measures(seg_bnd_tp, seg_bnd_fp, seg_bnd_fn,
                                                         scope="seg_bnd", penalty=mdl.data.batch_size)

        sums = [tf.summary.scalar(f'{name}/seg/precision', self.seg_tok_metric.precision, []),
                tf.summary.scalar(f'{name}/seg/recall', self.seg_tok_metric.recall, []),
                tf.summary.scalar(f'{name}/seg/f1', self.seg_tok_metric.f1, []),
                tf.summary.scalar(f'{name}/seg_bnd/f1', self.seg_bnd_metric.f1, [])]

        for tag in mdl.taggers.values():
            accuracy = self.metrics.accuracy(tag.mask, tag.hits, tag.name)
            sums.append(tf.summary.scalar(f'{name}/{tag.name}/acc', accuracy, []))

        self.summaries = tf.summary.merge(sums)

    def run(self, step, emit=True):
        sess = self.ma.sess
        sess.run([self.iterator.initializer, self.metrics.resets])
        feed_dict = self.mdl.data.feed_dict(sess, self.iterator)
        try:
            while True:
                sess.run(self.metrics.updates, feed_dict=feed_dict)
        except tf.errors.OutOfRangeError:
            if emit:
                summaries = sess.run(self.summaries)
                self.ma.summary_writer.add_summary(summaries, step)
            if self.debug:
                metric = self.seg_tok_metric
                data = sess.run([metric.tp, metric.fp, metric.fn])
                print(f"{self.name} seg: {data[0]} {data[1]} {data[2]} {data[0] + data[2]}")


class TrainExecutor(object):
    def __init__(self, ma: 'MorphAnalyzer'):
        self.ma = ma
        self.train_iter = ma.train_pipe.pipeline.make_one_shot_iterator()
        self.train_feed = ma.input_data.feed_dict(ma.sess, self.train_iter)

        tcfg = ma.model.cfg.get_config('times', Config())

        self.timers = [
            TimedExecution(tcfg.get_float('summary', 2.0), self.save_summaries, tf.summary.merge_all()),
            TimedExecution(tcfg.get_float('snapshot', 120.0), self.save_snapshot)
        ]

        evcfg = ma.model.cfg.get_config('eval', Config())
        for name, cfgname in evcfg.items():
            xcfg = ma.model.cfg.get_config(cfgname)
            period = xcfg.get_float('period', None)
            if period is not None:
                evobj = Evaluator(xcfg, ma, name, ma.infer_model())
                timer = TimedExecution(period, evobj.run)
                self.timers.append(timer)
                print(f"Registered evaluator {name}, it will run once per {period} seconds", file=sys.stderr)
            else:
                print(f"Evaluator {name} did not have period specified!", file=sys.stderr)

        self.timer_map = {}
        for i in range(len(self.timers)):
            self.timer_map[f'xtimer_{i}_key_impossible_to_forge!'] = self.timers[i]

    def save_summaries(self, step, summary_data):
        self.ma.summary_writer.add_summary(summary_data, step)

    def save_snapshot(self, step):
        ma = self.ma
        ma.saver.save(ma.sess, ma.snap_dir + '/snap', step)

    def execute(self):
        sess = self.ma.sess

        global_step = sess.run(self.ma.model.global_step)
        trace_steps = self.ma.model.cfg.get_list('trace_steps', [])
        if trace_steps:
            print("Tracing on", trace_steps, global_step, file=sys.stderr)
            if global_step == 0:
                self.ma.summary_writer.add_graph(self.ma.graph)

        start_examples = sess.run(self.ma.model.num_sentences)
        max_examples = self.ma.model.cfg.get_int('max_examples', None)

        while True:
            run_conf = tf.RunOptions()

            trace_now = global_step in trace_steps
            if trace_now:
                run_conf.trace_level = tf.RunOptions.FULL_TRACE
                run_metadata = tf.RunMetadata()
            else:
                run_metadata = None

            fetches = {
                'train': self.ma.optimizer.opt_op,
                'step': self.ma.model.inc_num_sentences,
                'train_avg': self.ma.train_summaries.update_avg,
                'global_step': self.ma.model.global_step
            }

            now = time.monotonic()
            for tkey, t in self.timer_map.items():
                if t.should_run(now):
                    fetches[tkey] = t.fetches

            results = sess.run(fetches,
                               feed_dict=self.train_feed,
                               options=run_conf,
                               run_metadata=run_metadata)

            step = results['step']
            global_step = results['global_step']

            for o in results:
                t = self.timer_map.get(o, None)
                if t is not None:
                    t.callback(step, *results[o])
                    t.last_run = time.monotonic()

            if trace_now:
                self.ma.summary_writer.add_run_metadata(run_metadata, str(step), global_step)

            if max_examples is not None:
                num_examples = step - start_examples
                if num_examples > max_examples:
                    raise tf.errors.OutOfRangeError()


class ValidOnlyExecutor(object):
    def __init__(self, ma):
        self.ma = ma
        self.validators = []

        evcfg = ma.model.cfg.get_config('eval', Config())
        for name, cfgname in evcfg.items():
            xcfg = ma.model.cfg.get_config(cfgname)
            self.validators.append(Evaluator(xcfg, ma, name, ma.infer_model()))

    def run(self):
        step = self.ma.sess.run(self.ma.model.num_sentences)
        for v in self.validators:
            v.debug = True
            print("Running", v.name, "at step", step, file=sys.stderr)
            v.run(step, emit=False)


class MorphAnalyzer(object):
    def __init__(self, cfg: Config):
        tag_def = cfg.get_list('tags')
        self.num_tags = cfg.get_int('num_tags', len(tag_def))
        self.is_infer = cfg.get('mode', None) == "infer"
        self.infer_mdl_ = None

        self.bert = cfg.get('bert', None)

        self.graph = tf.Graph()

        with self.graph.as_default():
            self.graph.seed = cfg.get_int('seed', 0xdeadbeef)
            has_bert = self.bert is not None and not self.is_infer
            self.train_pipe = InputPipeline(cfg.get_config('input.train'), self.num_tags, has_bert)
            self.input_data = InputData(self.num_tags, self.train_pipe.pipeline, self.is_infer, has_bert)
            self.dics = Dictionaries(cfg.get_config('dics'))
            with tf.variable_scope("ma_model", reuse=False):
                self.model = MorphModel(cfg, self.input_data, self.dics, self.is_infer, self.bert)
                layer_cfg = self.model.cfg.get_list('layers')
                tag_cfg = self.model.cfg.get_list('tags')
                self.model.initialize(layer_cfg, tag_cfg)

            self.optimizer = Optimizer(cfg, self.model)

            self.snap_dir = cfg.get_string('full_snapshot_dir')
            self.summary_writer = tf.summary.FileWriter(
                logdir=self.snap_dir
            )

            self.train_summaries = TrainSummaries(self)

            sess_bldr = tf.train.SessionManager(
                local_init_op=tf.local_variables_initializer(),
                graph=self.graph
            )

            self.saver = tf.train.Saver()

            snap_dir_passed = self.snap_dir
            snap_file_passed = None

            cpt_offset = cfg.get_int('checkpt_offset', None)

            if self.is_infer and cpt_offset is not None:
                self.saver.recover_last_checkpoints([self.snap_dir])
                snap_state = tf.train.get_checkpoint_state(self.snap_dir)
                all_paths = snap_state.all_model_checkpoint_paths
                snap_path = all_paths[cpt_offset]
                snap_dir_passed = None
                snap_file_passed = snap_path
                print("Using snapshot:", snap_path, file=sys.stderr)

            self.sess = sess_bldr.prepare_session(
                master=cfg.get_string('master', ''),
                init_op=tf.global_variables_initializer(),
                saver=self.saver,
                checkpoint_dir=snap_dir_passed,
                checkpoint_filename_with_path=snap_file_passed,
                config=self.make_config(cfg)
            )

            seedcfg = cfg.get_config('model_seed', Config())
            if seedcfg and len(self.saver.last_checkpoints) == 0:
                self._import_from_seed(seedcfg)

            self.dics.init_lookups(self.sess)

    def _import_from_seed(self, cfg: Config):
        path = cfg.get_string('path', None)
        if path is None:
            cpt_dir = cfg.get_string('dir')
            cpt_offset = cfg.get_int('offset', -1)
            snap_state = tf.train.get_checkpoint_state(cpt_dir)
            path = snap_state.all_model_checkpoint_paths[cpt_offset]

        print("loading seed model from", path, file=sys.stderr)
        rdr = tf.train.NewCheckpointReader(path)

        curvars = tf.get_variable_scope().trainable_variables()
        curvars = dict((x.name[:-2], x) for x in curvars)

        filters = cfg.get_list('filters', [])
        filter_res = [re.compile(pat) for pat in filters]

        def check_name(x):
            for flt in filter_res:
                if flt.fullmatch(x):
                    return False
            return True

        to_restore = []
        for name in rdr.get_variable_to_dtype_map():
            if name in curvars and "Adam" not in name:
                if check_name(name):
                    to_restore.append(curvars[name])
                else:
                    print("tensor", name, "was filtered out", file=sys.stderr)

        saver = tf.train.Saver(to_restore)
        saver.restore(self.sess, path)

    def infer_model(self):
        if self.infer_mdl_ is not None:
            return self.infer_mdl_
        if self.model.infer:
            return self.model

        with self.graph.as_default():
            with tf.variable_scope("ma_model", reuse=True):
                mdl: MorphModel = MorphModel(self.model.cfg, self.model.data, self.model.dicts, True, self.bert)
                layer_cfg = self.model.cfg.get_list('layers')
                tag_cfg = self.model.cfg.get_list('tags')
                mdl.initialize(layer_cfg, tag_cfg)

        self.infer_mdl_ = mdl
        return mdl

    def make_config(self, cfg: Config):
        c = cfg.get_config('session', Config())

        opts = tf.ConfigProto()
        opts.gpu_options.per_process_gpu_memory_fraction = c.get_float('gpu.mem_frac', 0)
        opts.gpu_options.allow_growth = c.get_bool('gpu.growth', False)

        opt_opts = opts.graph_options.optimizer_options
        if c.get_bool('use_xla', False):
            opt_opts.global_jit_level = tf.OptimizerOptions.ON_1

        opts.inter_op_parallelism_threads = c.get_int('nthreads_inter', 0)
        opts.intra_op_parallelism_threads = c.get_int('nthreads_intra', 0)

        return opts

    def inference(self):
        from xmrph.inference import Inference
        return Inference(self)

    def valid(self):
        with self.graph.as_default():
            vrun = ValidOnlyExecutor(self)
            vrun.run()

    def train(self):
        with self.graph.as_default():
            try:
                os.makedirs(self.snap_dir, exist_ok=True)
                trainer = TrainExecutor(self)
                trainer.execute()
            except tf.errors.OutOfRangeError:
                self.saver.save(
                    sess=self.sess,
                    save_path=self.snap_dir + "/snap",
                    global_step=self.model.num_sentences
                )


def main():
    cfg = parse_config_args()
    c2c = MorphAnalyzer(cfg)

    mode = cfg.get_string('mode', 'train')
    if mode == "train":
        curnice = os.nice(0)
        if curnice == 0:
            os.nice(cfg.get_int('nice', 5))
        c2c.train()
    elif mode == "infer":
        inf = c2c.inference()
        inf.run(cfg.get_string('infer'))
    elif mode == "valid":
        c2c.valid()
    else:
        print("non-supported mode", mode, file=sys.stderr)
        exit(1)


if __name__ == '__main__':
    np.set_printoptions(
        precision=3,
        linewidth=150
    )
    main()
