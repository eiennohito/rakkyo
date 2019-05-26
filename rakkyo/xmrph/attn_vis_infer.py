import matplotlib.figure as mpf
import matplotlib.lines as lines
import matplotlib.cm as mplcm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import os
from .layers import TransformerLayer
from .infer_base import BasicInferrer

from matplotlib import rcParams

rcParams['savefig.pad_inches'] = 0


class TransformerVis(object):
    def __init__(self, name, font_path):
        self.name = name
        self.font = FontProperties(
            fname=font_path
        )

    def draw(self, viz, pdf: PdfPages, chars: str, attn_all):
        idx = viz.idx
        # attn is b x h x l x l
        attn = attn_all[idx]
        num_heads = attn.shape[0]
        text_len = len(chars) + 2
        fig = mpf.Figure(figsize=(text_len * 0.5, 2 * num_heads), tight_layout=True)

        cmap = mplcm.get_cmap('pink')

        display_border = 0.3 / text_len

        for head in range(num_heads):
            ax = fig.add_subplot(num_heads, 1, 1 + head, xlim=(-1, text_len + 1), frameon=False)
            attn_matrix = attn[head]
            for i, c in enumerate(chars):
                ax.text(1 + i, 0.9, c, fontproperties=self.font)
                ax.text(1 + i, 0.1, c, fontproperties=self.font)
            ax.text(0, 0.9, 'B', fontproperties=self.font)
            ax.text(0, 0.1, 'B', fontproperties=self.font)
            ax.text(len(chars) + 1, 0.9, 'E', fontproperties=self.font)
            ax.text(len(chars) + 1, 0.1, 'E', fontproperties=self.font)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            colors = cmap(1 - attn_matrix)

            for i in range(text_len):
                for j in range(text_len):
                    v = attn_matrix[i, j]
                    if v > display_border:
                        line = lines.Line2D([i, j], [0.2, 0.8], color=colors[i, j])
                        ax.add_line(line)
        pdf.attach_note(self.name)
        pdf.savefig(fig)


class AttentionViz(BasicInferrer):
    def __init__(self, ma):
        super().__init__(ma)
        self.processed = 0
        self.base = ma.model.cfg.get_string('base_name')
        font_path = ma.model.cfg.get_string('font_path')

        drawing = {}

        mdl = ma.model
        for l in mdl.layers:
            if isinstance(l, TransformerLayer):
                l: TransformerLayer = l
                attn = l.impl.attn.Attn
                drawer = TransformerVis(l.name, font_path)
                drawing[drawer] = attn

        self.fetches['drawing'] = drawing

    def format_result(self, writer, comment, data, raw_tags):
        name = "%s-%05d.pdf" % (self.base, self.processed)
        basename = os.path.dirname(name)
        os.makedirs(basename, exist_ok=True)
        self.processed += 1

        with PdfPages(name) as pdf:
            drawing = self.ctx['drawing']
            for drawer, tens in drawing.items():
                drawer.draw(self, pdf, data, tens)

            d = pdf.infodict()
            d['Title'] = comment
