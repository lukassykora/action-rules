"""Regenerate every empirical table and both statistical figures from archived CSVs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O = Path(__file__).resolve().parents[1]
R, D = O / 'results', O / 'manuscript'
T = D / 'tables'
T.mkdir(parents=True, exist_ok=True)
(D / 'figures').mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'font.family': 'sans-serif', 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.linewidth': .6})
names = {'point': 'Point', 'wald': 'Wald', 'wald_rd': 'Wald-RD', 'wilson': 'Newcombe--Wilson',
         'percentile': 'Percentile', 'bca': 'BCa', 'bayes': 'Bayes--uniform',
         'jeffreys': 'Bayes--Jeffreys', 'bayes_rd': 'Bayes--RD', 'cp': 'Simultaneous CP'}


def fmt_int(v):
    s = f'{abs(v):,.0f}'.replace(',', r'{,}')
    return f'$-{s}$' if round(v) < 0 else s


def fmt_ms(v):
    return f'{v:,.1f}'.replace(',', r'{,}')


def table(name, caption, label, spec, head, rows, size='scriptsize', preamble=''):
    text = (
        '\\begin{table}[tbp]\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\' + size + '\n'
        + (preamble + '\n' if preamble else '')
        + '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}' + spec + '}\n\\toprule\n'
        + head + '\\\\\n\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular*}\n\\end{table}\n'
    )
    (T / name).write_text(text, encoding='utf-8')


e2 = pd.read_csv(R / 'e2_Attrition_summary.csv')


def val(flow, view, method, metric):
    a = e2.query('flow==@flow and view==@view and method==@method and metric==@metric')
    return a.iloc[0]['mean']


rows = []
for flow, view, desc, methods in [
    ('full', 'same', 'Full, same-sample; mean family 40.3; top-rule bias $+744$',
     ['point', 'wald', 'percentile', 'jeffreys', 'wilson']),
    ('split', 'same', 'Split, discovery half; mean family 86.4; top-rule bias $+1{,}443$',
     ['point', 'wald', 'wilson']),
    ('split', 'honest', 'Split, independent inference half; top-rule bias $+38$',
     ['point', 'wald', 'percentile', 'jeffreys', 'wilson', 'cp'])]:
    rows.append('\\multicolumn{7}{l}{\\emph{' + desc + '}}\\\\')
    fields = ['coverage', 'accept', 'abstain', 'false_accept', 'any_false_accept', 'top_coverage']
    for method in methods:
        cells = [val(flow, view, method, metric) for metric in fields]
        rows.append(names[method] + ' & ' + ' & '.join('--' if pd.isna(v) else f'{v:.3f}' for v in cells) + '\\\\')
table(
    'plasmode.tex',
    'E2 on Attrition: 1{,}000 replicates, empirical-population truth. Cov.: mean within-replicate coverage; '
    'Acc./Abst.: accepted / uncertain or unsupported fraction; FA: accepted fraction among nonpositive-true-gain rules '
    '(969 full and 990 split replicates contain such rules). Mean family size and Any FA use all 1{,}000 replicates; '
    'other rule summaries use 998 nonempty full or 1{,}000 split families. '
    'Top: coverage of the discovery-ranked top rule. Bias is in USD.',
    'tab:plasmode', 'lcccccc', 'Method & Cov. & Acc. & Abst. & FA & Any FA & Top', rows,
)

e3 = pd.read_csv(R / 'e3_intervals.csv')
rows = []
for method in ['point', 'wald', 'wald_rd', 'wilson', 'percentile', 'bca', 'bayes', 'jeffreys', 'cp']:
    counts = []
    for target in ['G', 'U']:
        sub = e3.query('dataset=="Attrition" and method==@method and target==@target').category.value_counts()
        counts.append(' / '.join(str(sub.get(k, 0)) for k in ['accept', 'uncertain', 'reject']))
    rows.append(names[method] + ' & ' + ' & '.join(counts) + '\\\\')
table(
    'decisions.tex',
    'E3 same-sample decisions for 21 Attrition rules (above / uncertain / below, threshold $\\tau=0$; no unsupported '
    'intervals). Percentile, BCa ($B=999$, seed 42), Wald-RD, and Simultaneous CP use the random design; Wald, '
    'Newcombe--Wilson, and both Beta--Binomial posteriors condition on observed prevalence.',
    'tab:decisions', 'lcc', 'Method & $G$ & $U$', rows,
)

sens = pd.read_csv(R / 'e3_sensitivity.csv')
rows = []
for kind, value, desc in [
    ('baseline', '1.0', 'Baseline settings'),
    ('cost', '0.5', r'Costs $\times0.5$'),
    ('cost', '1.5', r'Costs $\times1.5$'),
    ('support', str(22 / 18), 'Support 22/22'),
    ('support', str(14 / 18), 'Support 14/14'),
    ('actionable', '1.0', 'Flexible: OverTime, Income'),
]:
    sub = sens[(sens.dataset == 'Attrition') & (sens.type == kind) &
               np.isclose(pd.to_numeric(sens.setting, errors='coerce'), float(value))]
    counts = []
    for method in ['point', 'wilson', 'cp']:
        row = sub[sub.method == method].iloc[0]
        counts.append(' / '.join(str(int(row[k])) for k in ['accept', 'uncertain', 'reject']))
    rows.append(desc + f' & {int(row.rules)} & ' + ' & '.join(counts) + '\\\\')
table(
    'sensitivity.tex',
    'Attrition sensitivity for $G$ (above / uncertain / below). Cost variations scale every negative utility entry '
    '(including the negative intrinsic utility of \\texttt{OverTime=Yes}); '
    'support gives the undesired/desired absolute count thresholds. NW: Newcombe--Wilson.',
    'tab:sensitivity', 'lrccc', 'Variation & Rules & Point & NW & CP', rows,
)

cv = pd.read_csv(R / 'e4_fold_results.csv')
wide = cv.query('dataset=="Attrition"').pivot(index=['repeat', 'fold'], columns='strategy', values='mean_gain')
rows = []
for strategy, name in [
    ('point_U', r'Point $U$'),
    ('lower_U', r'Lower bound of $U$'),
    ('point_G', r'Point $G$'),
    ('lower_G', r'Lower bound of $G$'),
    ('filter_G', r'Lower $G$, filter $>0$'),
]:
    values = [cv.query('dataset==@ds and strategy==@strategy').mean_gain.mean()
              for ds in ['Telco', 'Bank', 'Attrition', 'Credit']]
    diff = wide[strategy] - wide.point_G
    pair = (f'{(diff > 1e-8).mean():.2f} / {(diff < -1e-8).mean():.2f}' if strategy != 'point_G' else '-- / --')
    rows.append(name + ' & ' + ' & '.join(fmt_int(v) for v in values) +
                f' & {pair} / {(wide[strategy] < 0).mean():.2f}' + '\\\\')
table(
    'cv.tex',
    'E4: equal-rule mean held-out gain (USD) of the top 20\\% eligible rules ($10\\times5$ folds). '
    'Attrition columns also report paired better/worse fractions against point-$G$ ranking (ties occupy the remainder) '
    'and the fraction of splits with negative test means, omitting unavailable test estimates.',
    'tab:cv', 'lrrrrc', 'Ranking by & Telco & Bank & Attr. & Credit & Attr.: better/worse/neg.', rows,
)

runtime = pd.read_csv(R / 'e5_runtime.csv')
rt_methods = [
    ('counting', 'Counting only'),
    ('wald', 'Wald'),
    ('wilson', 'Newcombe--Wilson'),
    ('percentile', 'Percentile'),
    ('bca', 'BCa'),
    ('jeffreys', 'Bayes--Jeffreys'),
    ('cp', 'Simultaneous CP'),
    ('row_api', 'Legacy row bootstrap'),
    ('nw_api', 'Legacy analytic NW'),
]
rows = []
for idx, (method, label) in enumerate(rt_methods):
    cells = []
    for ds in ['Telco', 'Bank', 'Attrition', 'Credit']:
        s = runtime.query('dataset==@ds and method==@method').seconds * 1000
        cells.append(r'\rtcell{' + fmt_ms(s.median()) + '}{' + f'{fmt_ms(s.min())}, {fmt_ms(s.max())}' + '}')
    suffix = r'\\[1.5pt]' if idx < len(rt_methods) - 1 else r'\\'
    rows.append(label + ' & ' + ' & '.join(cells) + suffix)
table(
    'runtime.tex',
    'E5 median and [min, max] run times in milliseconds per complete rule family (7 serial repetitions after one '
    'warm-up; Apple M5 Pro, macOS 26.6.2, Python 3.12.6, NumPy 1.26.4, SciPy 1.13.1). All inference rows include '
    'counting ($B=999$ for bootstrap, $M=10{,}000$ for posterior). Count-level methods return $G,U,H$; legacy engines '
    'return $G,U$.',
    'tab:runtime', 'lrrrr', 'Method & Telco & Bank & Attrition & Credit', rows,
    preamble=r'\newcommand{\rtcell}[2]{\begin{tabular}[t]{@{}r@{}}#1\\[-1pt]{\tiny[#2]}\end{tabular}}',
)

e1 = pd.read_csv(R / 'e1_summary.csv', keep_default_na=False)
methods = ['wald', 'wilson', 'percentile', 'bca', 'bayes', 'jeffreys', 'wald_rd', 'bayes_rd', 'cp']
short = ['Wald', 'NW', 'Percentile', 'BCa', 'Uniform', 'Jeffreys', 'Wald-RD', 'Bayes-RD', 'CP']
cols = [(s, e) for s in ['rich', 'moderate', 'sparse'] for e in ['null', 'modest', 'strong', 'boundary']]
fig, axes = plt.subplots(2, 1, figsize=(5.3, 5.25), layout='constrained', gridspec_kw={'height_ratios': [7, 9]})
for ax, (design, target, label) in zip(
    axes, [('fixed', 'G', '(a) G: fixed design'), ('random', 'U', '(b) U: random design')]
):
    sub = e1.query('design==@design and target==@target').set_index(['method', 'support', 'effect'])
    shown = [m for m in methods if design != 'fixed' or m not in ['wald_rd', 'bayes_rd']]
    labels = [short[methods.index(m)] for m in shown]
    mat = np.array([[sub.loc[(m, s, e), 'coverage'] for s, e in cols] for m in shown])
    ax.imshow(mat, vmin=.2, vmax=1, cmap='Greys_r', aspect='auto')
    ax.set_yticks(range(len(shown)), labels)
    ax.set_xticks(range(12), ['N', 'M', 'S', 'B'] * 3)
    ax.tick_params(length=0, pad=3)
    for i, m in enumerate(shown):
        for j, (s, e) in enumerate(cols):
            row = sub.loc[(m, s, e)]
            star = '*' if row.unsupported / row.replicates > .05 else ''
            ax.text(j, i, f'{mat[i, j]:.2f}'.lstrip('0') + star, ha='center', va='center',
                    fontsize=8, color='white' if mat[i, j] < .6 else 'black')
    for j, support in zip([1.5, 5.5, 9.5], ['Rich', 'Moderate', 'Sparse']):
        ax.text(j, -.75, support, ha='center', va='bottom', clip_on=False)
    ax.text(0, 1.15, label, transform=ax.transAxes, ha='left', va='bottom', fontweight='bold')
    for x in [3.5, 7.5]:
        ax.axvline(x, color='black', linewidth=.8)
    ax.set_xlim(-.5, 11.5)
    ax.set_ylim(len(shown) - .5, -.5)
fig.savefig(D / 'figures/Fig2.pdf', bbox_inches='tight')
plt.close(fig)

details = pd.read_csv(R / 'e3_rule_details.csv').query('dataset=="Attrition"').set_index('rule')
sub = e3.query('dataset=="Attrition" and target=="G"')
nw = sub.query('method=="wilson"').set_index('rule').sort_values('point', ascending=False)
cp = sub.query('method=="cp"').set_index('rule').loc[nw.index]

val_map = {'Research & Development': 'R&D', 'medium': 'med', 'very_high': 'v.high'}
key_map = {'OverTime': 'OT', 'MonthlyIncome': 'Inc', 'JobSatisfaction': 'JobSat',
           'EnvironmentSatisfaction': 'EnvSat', 'WorkLifeBalance': 'WLB'}
labels = []
for rule in nw.index:
    d = details.loc[rule]
    u, v = json.loads(d.mask_u), json.loads(d.mask_d)
    fixed, change = [], []
    for key in u:
        uval = val_map.get(u[key], u[key])
        vval = val_map.get(v.get(key, ''), v.get(key, ''))
        if u[key] == v.get(key):
            if key in ('Age', 'Gender', 'MaritalStatus', 'Department'):
                fixed.append(uval)
            else:
                fixed.append(f'{key_map.get(key, key)}={uval}')
        else:
            change.append(f'{key_map.get(key, key)}: {uval}→{vval}')
    labels.append(f'R{rule} ({", ".join(fixed)}) ' + ', '.join(change))

fig, ax = plt.subplots(figsize=(6.4, 5.1), layout='constrained')
y = np.arange(len(nw))
for yi in y:
    if yi % 2 == 1:
        ax.axhspan(yi - 0.5, yi + 0.5, color='#F7F7F7', zorder=0)
ax.hlines(y, cp.lower, cp.upper, color='.65', linewidth=2.6, label='Family CP', zorder=2)
ax.hlines(y, nw.lower, nw.upper, color='black', linewidth=1.1, label='Marginal NW', zorder=3)
ax.plot(nw.point, y, 'o', color='black', markersize=3, label='Point gain', zorder=4)
ax.axvline(0, color='black', ls='--', linewidth=.7, zorder=1)
ax.set_yticks(y, labels, fontsize=7.5)
ax.invert_yaxis()
ax.set_xlabel('Modeled gain G (USD per matched employee)')
ax.tick_params(axis='y', length=0)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.legend(loc='lower right', fontsize=7.5, frameon=True)
ax.grid(axis='x', color='.9', linewidth=.5, zorder=0)
fig.savefig(D / 'figures/Fig3.pdf', bbox_inches='tight')
plt.close(fig)
print('Generated five tables and two vector figures from archived results.')
