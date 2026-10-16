import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os, sys
LANG = os.environ.get('P2_LANG', 'en')
L = lambda i, e: e if LANG == 'en' else i
INK, MUTED = '#2b2a27', '#6b6a64'
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, ax = plt.subplots(figsize=(7.4, 3.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 52); ax.axis('off')
boxes = [
 (1.5, L('Komposisi\nkelas', 'Class\ncomposition'), [L('n minoritas', 'n minority'), L('n mayoritas', 'n majority'), L('(Seri A / Seri B)', '(Series A / B)')], '#f1efe8', '#898781'),
 (21.5, L('Noise & sinyal\nevaluasi', 'Evaluation\nnoise & signal'), [L('σ noise, σ sinyal', 'σ noise, σ signal'), 'SNR, ICC, τ'], '#e6f0fc', '#2a78d6'),
 (41.5, L('Landscape\nterukur', 'Measured\nlandscape'), ['FDC, dispersion', L('ρ₁, neutralitas', 'ρ₁, neutrality'), L('R² surrogate,', 'surrogate R²,'), L('kepentingan HP', 'HP importance')], '#e6f0fc', '#2a78d6'),
 (61.5, L('Perilaku\npencarian', 'Search\nbehaviour'), ['diversity', L('tarikan ke batas', 'pull to bounds'), 'RS, BFO, HHO,', 'SMA, SMAc'], '#fdeee6', '#eb6834'),
 (81.5, L('Keandalan\noptimasi', 'Optimization\nreliability'), ["winner's curse", L('evaluasi ulang', 're-evaluation'), 'vs RS & default'], '#e5f6ef', '#1baf7a'),
]
W, H, Y = 16.5, 25, 12
for x, t, items, fc, ec in boxes:
    ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle='round,pad=0.4,rounding_size=1.5', fc=fc, ec=ec, lw=1.4))
    ax.text(x + W / 2, Y + H - 2.2, t, ha='center', va='top', fontsize=7.9, fontweight='bold', color=INK, linespacing=1.1)
    for i, it in enumerate(items):
        ax.text(x + W / 2, Y + H - 10.2 - i * 3.1, it, ha='center', va='top', fontsize=7, color=INK)
for x in (18.3, 38.3, 58.3, 78.3):
    ax.add_patch(FancyArrowPatch((x, Y + H / 2), (x + 2.4, Y + H / 2), arrowstyle='-|>', mutation_scale=11, color=INK, lw=1.3))
# manipulasi (atas)
def tag(x0, x1, y, text, col):
    ax.annotate('', xy=(x1, Y + H + 0.6), xytext=(x1, y - 1.2), arrowprops=dict(arrowstyle='-|>', color=col, lw=1.1))
    ax.text(x0, y, text, ha='center', va='bottom', fontsize=7, linespacing=1.15, color=col, fontweight='bold')
tag(9.75, 9.75, 42, L('Manipulasi:\nsubsampling ke level IR', 'Manipulation:\nsubsampling to IR levels'), MUTED)
tag(29.75, 29.75, 42, L('Manipulasi:\ncv5x1 vs cv5x3', 'Manipulation:\ncv5x1 vs cv5x3'), MUTED)
tag(69.75, 69.75, 42, L('Manipulasi:\nflip test (x → 1 − x)', 'Manipulation:\nflip test (x → 1 − x)'), MUTED)
# RQ (bawah)
for x0, x1, lab in [(1, 39, 'RQ1'), (41.5, 59, 'RQ2'), (61.5, 99, 'RQ3')]:
    ax.plot([x0 + .5, x1 - .5], [8.5, 8.5], color=INK, lw=1)
    for xx in (x0 + .5, x1 - .5):
        ax.plot([xx, xx], [8.5, 10], color=INK, lw=1)
    ax.text((x0 + x1) / 2, 6.8, lab, ha='center', va='top', fontsize=8.5, fontweight='bold', color=INK)
ax.text(50, 1.0, L('280 unit = 14 kondisi komposisi kelas × 2 skema CV × 10 run; fitness = rata-rata weighted F1 pada stratified 5-fold CV', '280 units = 14 class-composition conditions × 2 CV schemes × 10 runs; fitness = mean weighted F1 over stratified 5-fold CV'),
        ha='center', va='bottom', fontsize=7, color=MUTED)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures', L('Gambar0_kerangka.png', 'Fig1_framework.png'))
fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05)
fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight', pad_inches=0.05)
