class BasicInferrer(object):
    def __init__(self, ma):
        self.ma = ma
        self.tag_tensors, self.tag_lookups = ma.model.tag_lookups(
            'seg', 'pos', 'subpos', 'ctype', 'cform'
        )
        self.fetches = dict(ma.model.data.next)
        self.fetches['tags'] = self.tag_tensors
        self.idx = 0
        self.ctx = {}

    def run_inference(self, feed_dict, writer):
        elem = self.ma.sess.run(
            self.fetches,
            feed_dict=feed_dict
        )

        comment = elem['comment']
        data = elem['raw']
        tags = elem['tags']

        xlen = comment.shape[0]
        self.ctx = elem
        for i in range(xlen):
            self.idx = 0
            self.format_result(writer, comment[i], data[i].decode('utf-8'), tags[i])

    def format_result(self, writer, comment, data, raw_tags):
        pass
