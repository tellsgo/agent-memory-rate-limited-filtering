"""Shared publication style and reproducible export checks for the three figures."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image

# Use installed TrueType Times faces; STIX is the compatible Times-style math face.
for path in Path('/System/Library/Fonts/Supplemental').glob('Times New Roman*.ttf'):
    font_manager.fontManager.addfont(str(path))
font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 8,
    'axes.titlesize': 8, 'axes.linewidth': .8, 'axes.edgecolor': '#000000',
    'lines.linewidth': 1., 'lines.markersize': 3.5,
    'lines.markeredgewidth': .7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.frameon': False, 'legend.handlelength': 2.,
    'legend.handletextpad': .45, 'legend.columnspacing': 1.,
    'legend.borderaxespad': 0., 'legend.borderpad': 0.,
})
COLORS = {'qwen2.5-7b': '#E69F00', 'qwen2.5-14b': '#004E82',
          'phi-4': '#D55E00', 'qwen2.5-72b': '#008862', 'gpt-oss-120b': '#CC79A7'}
MARKERS = {'qwen2.5-7b': 'o', 'qwen2.5-14b': 's', 'phi-4': '^',
           'qwen2.5-72b': 'D', 'gpt-oss-120b': 'v'}
MARKER_SIZES = {'qwen2.5-7b': 5.5, 'qwen2.5-14b': 5.5, 'phi-4': 6.8,
                'qwen2.5-72b': 5.8, 'gpt-oss-120b': 6.5}
NAMES = {r: r.replace('qwen', 'Qwen').replace('b', 'B') if r.startswith('qwen') else r for r in COLORS}

def read(name):
    with (ROOT.parent / 'figdata' / name).open(newline='') as f:
        return list(csv.DictReader(f))

def values(rows, col):
    return np.array([float(r[col]) for r in rows])

def subset(rows, **filters):
    return [r for r in rows if all(r[k] == str(v) for k, v in filters.items())]

def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#e6e6e6', linewidth=.5)
    ax.tick_params(pad=2)

def log_ticks(ax, ticks):
    ax.set_xscale('log', base=2)
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.0f}'))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

def panel(ax, letter):
    ax.text(0, 1.045, f'({letter})', transform=ax.transAxes,
            weight='bold', va='bottom', ha='left', fontsize=8)

def reader_handle(reader):
    return Line2D([], [], color=COLORS[reader], marker=MARKERS[reader],
                  linestyle='none', markersize=MARKER_SIZES[reader],
                  markeredgewidth=.7, label=NAMES[reader])

def errorbar(ax, x, y, lo, hi, **kwargs):
    assert np.all(lo <= y) and np.all(y <= hi)
    result = ax.errorbar(x, y, yerr=np.vstack((y-lo, hi-y)),
                        **(dict(capsize=1.6, capthick=.65, elinewidth=.65) | kwargs))
    # Axis limits enclose all CIs. Keep endpoint glyphs intact at exact boundaries
    # (e.g. accuracy 1), without permitting any interval to breach the frame.
    assert np.all(lo >= ax.get_ylim()[0]) and np.all(hi <= ax.get_ylim()[1])
    for artist in [result.lines[0], *result.lines[1], *result.lines[2]]:
        if artist is not None:
            artist.set_clip_on(False)
    return result

def export(fig, stem, counts):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = []
    for t in texts:
        box = t.get_window_extent(renderer)
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            outside.append(t.get_text())
    assert not outside, f'Text outside canvas: {outside}'
    assert min(t.get_fontsize() for t in texts) >= 7
    # Never bbox_inches='tight': retain the exact physical dimensions in the brief.
    fig.savefig(ROOT / f'{stem}.pdf')
    fig.savefig(ROOT / f'{stem}.png', dpi=300)
    checkdir = ROOT / 'checks'
    checkdir.mkdir(exist_ok=True)
    with Image.open(ROOT / f'{stem}.png') as im:
        im.convert('L').save(checkdir / f'{stem}_gray.png', dpi=(300,300))
        pixels = list(im.size)
    report = {'size_inches': list(fig.get_size_inches()), 'png_pixels': pixels,
              'minimum_font_pt': min(t.get_fontsize() for t in texts),
              'text_outside_canvas': outside, 'counts': counts,
              'font': font_manager.findfont('Times New Roman'), 'pdf_fonttype': 42}
    (checkdir / f'{stem}.json').write_text(json.dumps(report, indent=2)+'\n')
