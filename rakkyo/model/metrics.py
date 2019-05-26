import tensorflow as tf
import time


class Windowed(object):
    def __init__(self, x, window, step, name):
        xshape = x.shape
        xshape = tf.TensorShape([window]).concatenate(xshape)
        ctr = tf.mod(step, window)
        self.storage = tf.get_local_variable(
            name=name,
            shape=xshape,
            dtype=x.dtype,
            initializer=tf.zeros_initializer(dtype=x.dtype),
            trainable=False
        )
        self.update = tf.scatter_update(
            ref=self.storage,
            indices=ctr,
            updates=x
        )
        self.mean_readonly = tf.reduce_mean(self.storage, axis=0)
        self.mean_updated = tf.tuple([self.mean_readonly], control_inputs=[self.update])[0]

    def maybe_gated(self, gated):
        if gated:
            return self.update
        else:
            return self.storage

    def sum(self, gated=False):
        return tf.reduce_sum(self.maybe_gated(gated), 0)

    def mean(self, gated=False):
        return tf.reduce_mean(self.maybe_gated(gated), 0)


class RateCalculator(object):
    def __init__(self, value, window, step, name="rate"):
        ts = tf.timestamp()
        self.prev_ts = tf.get_local_variable(
            name=f'{name}/prev_ts',
            shape=(),
            dtype=ts.dtype,
            initializer=tf.constant_initializer(time.time(), ts.dtype)
        )
        diff = ts - self.prev_ts
        with tf.control_dependencies([diff]):
            self.update_ts = self.prev_ts.assign(ts)

        self.values = Windowed(value, window, step, f'{name}/values')
        self.times = Windowed(tf.to_float(diff), window, step, f'{name}/times')
        self.updates = [
            self.update_ts,
            self.values.update,
            self.times.update
        ]

    def rate(self, gated=True):
        xvalues = tf.to_float(self.values.sum(gated))
        xtimes = self.times.sum(gated)
        return tf.where(tf.not_equal(xtimes, 0), xvalues / xtimes, tf.to_float(0))


class MatchEvaluation(object):
    def __init__(self):
        self.tp = tf.get_local_variable(
            name='mtch/tp',
            dtype=tf.int64,
            shape=(),
            trainable=False
        )

        self.fp = tf.get_local_variable(
            name='mtch/fp',
            dtype=tf.int64,
            shape=(),
            trainable=False
        )

        self.fn = tf.get_local_variable(
            name='mtch/fn',
            dtype=tf.int64,
            shape=(),
            trainable=False
        )

        f_tp = tf.maximum(tf.to_float(self.tp), 0)
        f_fp = tf.to_float(self.fp)
        f_fn = tf.to_float(self.fn)

        self.precision = f_tp / tf.maximum(f_tp + f_fp, 1.0)
        self.recall = f_tp / tf.maximum(f_tp + f_fn, 1.0)
        self.f1 = 2 * self.precision * self.recall / tf.maximum(self.precision + self.recall, 1e-9)


def maybe_toint32(x):
    if x.dtype == tf.bool:
        x = tf.to_int32(x)
    shp: tf.TensorShape = x.shape
    if shp.dims != 0:
        x = tf.reduce_sum(x)
    return x


class Metrics(object):
    def __init__(self, scope='reset_metrics'):
        self.scope = scope
        self.updates = []
        self.values = []
        self.resets = []
        self.summaries = []

    def xscope(self, scope):
        if scope is None:
            scope = self.scope
        else:
            scope = self.scope + "/" + scope
        return tf.variable_scope(scope)

    def make(self, metric, scope=None, **metric_args):
        with self.xscope(scope) as scope:
            metric_op, update_op = metric(**metric_args)
            vars = tf.contrib.framework.get_variables(
                scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
            reset_op = tf.variables_initializer(vars)
            self.updates.append(update_op)
            self.resets.append(reset_op)
            self.values.append(metric_op)
            return metric_op

    def accuracy(self, labels, predictions, scope=None):
        with self.xscope(scope) as scope:
            labels_cnt = tf.get_local_variable(
                name='acc/labels',
                shape=(),
                dtype=tf.int64,
                trainable=False
            )
            predictions_cnt = tf.get_local_variable(
                name='acc/predictions',
                shape=(),
                dtype=tf.int64,
                trainable=False
            )
            reset_op = tf.variables_initializer([labels_cnt, predictions_cnt])
            metric_op = tf.to_float(predictions_cnt) / tf.maximum(tf.to_float(labels_cnt), 1.0)
            update_op = tf.group([
                labels_cnt.assign_add(tf.to_int64(maybe_toint32(labels))),
                predictions_cnt.assign_add(tf.to_int64(maybe_toint32(predictions)))
            ])
            self.resets.append(reset_op)
            self.updates.append(update_op)
            self.values.append(metric_op)
            return metric_op

    def mean(self, values, scope=None):
        with self.xscope(scope) as scope:
            finite = tf.is_finite(values)
            finite_cnt = tf.reduce_sum(tf.to_int32(finite))
            sum = tf.get_local_variable(
                name='avg/sum',
                shape=(),
                dtype=tf.float32,
                trainable=False
            )
            count = tf.get_local_variable(
                name='avg/count',
                shape=(),
                dtype=tf.int64,
                trainable=False
            )
            reset_op = tf.variables_initializer([sum, count])
            metric_op = sum / tf.to_float(count + 1)
            update_op = tf.group([
                sum.assign_add(tf.reduce_sum(tf.boolean_mask(values, finite))),
                count.assign_add(tf.to_int64(finite_cnt))
            ])
            self.resets.append(reset_op)
            self.updates.append(update_op)
            self.values.append(metric_op)
            return metric_op

    def eval_measures(self, tp, fp, fn, scope=None, penalty=None):
        with self.xscope(scope) as scope:
            m = MatchEvaluation()
            reset_op = tf.variables_initializer([m.tp, m.fp, m.fn])
            tp_cnt = tf.to_int64(maybe_toint32(tp))
            if penalty is not None:
                tp_cnt = tp_cnt - tf.to_int64(penalty)
            update_op = tf.group([
                m.tp.assign_add(tp_cnt),
                m.fn.assign_add(tf.to_int64(maybe_toint32(fn))),
                m.fp.assign_add(tf.to_int64(maybe_toint32(fp)))
            ])
            self.resets.append(reset_op)
            self.updates.append(update_op)
            self.values.extend([m.tp, m.fp, m.fn])

            return m

    def eval_measures_den(self, tp, prec_den, rec_den, scope=None, penalty=None):
        with self.xscope(scope) as scope:
            m = MatchEvaluation()
            reset_op = tf.variables_initializer([m.tp, m.fp, m.fn])
            tp_cnt = tf.to_int64(maybe_toint32(tp))
            if penalty is not None:
                tp_cnt = tp_cnt - tf.to_int64(penalty)
            fn_cnt = tf.to_int64(maybe_toint32(rec_den)) - tp_cnt
            fp_cnt = tf.to_int64(maybe_toint32(prec_den)) - tp_cnt
            update_op = tf.group([
                m.tp.assign_add(tp_cnt),
                m.fn.assign_add(fn_cnt),
                m.fp.assign_add(fp_cnt)
            ])
            self.resets.append(reset_op)
            self.updates.append(update_op)
            self.values.extend([m.tp, m.fp, m.fn])

            return m

    def scalar(self, name, value):
        v = tf.summary.scalar(name, value, collections=[])
        self.summaries.append(v)
        return v

    def text(self, name, value, append=True):
        v = tf.summary.text(name, value, collections=[])
        if append:
            self.summaries.append(v)
        return v

    def reset(self):
        return tf.group(self.resets)

    def update(self):
        return tf.group(self.updates)

    def summary(self):
        return tf.summary.merge(self.summaries)
