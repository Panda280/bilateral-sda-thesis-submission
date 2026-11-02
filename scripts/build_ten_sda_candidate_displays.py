"""Render registered R-2026-051 results without recalculating SDA effects."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'implementation/bilateral_sda_attribution/docs/analysis/ten_sda_candidates_20260910'
OUT = ROOT/'src/chapters/results_revised/generated'
FIG = ROOT/'src/figures/results/ten_sda_candidates_20260910'
FIG.mkdir(parents=True, exist_ok=True)
s = pd.read_csv(BASE/'ten_candidates_validated.csv')
a = pd.read_csv(BASE/'selected_annual_paths.csv')
names = ['Japan: steel (43)', 'Japan: petroleum (32)', 'Italy: petroleum (32)',
         'Canada: steel (43)', 'Germany: petroleum (32)', 'Japan: rubber/plastics (37)',
         'Japan: fabricated metals (50)', 'Netherlands: petroleum (32)',
         'Japan: glass (38)', 'Czechia: office/computing (52)']
rows = []
for (_, r), name in zip(s.iterrows(), names):
    rows.append(f"{r.case_id} & {name} & {r.Home:+.2f} & {r.Foreign:+.2f} & {r.World:+.2f} & {r.persistent_01}/5 " + r'\\')
text = r'''\begin{table}[htbp]
\centering\small
\caption{Ten intermediate-sourcing candidates, P3, R3 grid (Mt CO$_2$).
Home and foreign identify emitters relative to the purchasing country;
foreign includes other Reduction countries. Codes identify source products,
not emitting industries. Annual counts apply the 0.1 Mt screen.}
\label{tab:ten-sda-candidates}
\begin{tabular}{llrrrr}
\toprule
Case & Purchaser: source product & Home & Foreign & World & Years\\
\midrule
''' + '\n'.join(rows) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
(OUT/'tab_ten_sda_candidates.tex').write_text(text)
rows=[]
for _,r in s.iterrows():
    years=', '.join(str(int(y)) for y in a.loc[(a.case_id==r.case_id)&a['year_pass_0.1'],'year'])
    rows.append(f"{r.case_id} & {r.Domestic:+.2f} & {r['Red']:+.2f} & {r['Non-Red']:+.2f} & {r['Non-Participant']:+.2f} " + r'\\')
(OUT/'tab_ten_sda_sources.tex').write_text(r'''\begin{table}[htbp]
\centering\small
\caption{Source-group partition of the same ten $\mathbf D_C$ slices (Mt CO$_2$).
All emitter locations are summed. Each row sums to its world contribution
in Table~\ref{tab:ten-sda-candidates}; the two tables must not be added.}
\label{tab:ten-sda-sources}
\begin{l*{4}{r}}
\end{l*{4}{r}}
\begin{tabular}{lrrrr}
\toprule
Case & Domestic & Foreign Red. & Non-Red. & Non-Participant\\
\midrule
''' .replace('\\begin{l*{4}{r}}\n\\end{l*{4}{r}}\n','')+'\n'.join(rows)+r'''
\bottomrule
\end{tabular}
\end{table}
''')
fig,ax=plt.subplots(figsize=(8.3,4.8))
y=list(range(len(s)))
ax.barh(y,s.ReductionEmitter,label='Reduction emitters',color='#21618c')
ax.barh(y,s.OutsideReductionEmitter,label='Emitters outside Reduction',color='#c66b27')
ax.scatter(s.World,y,marker='D',s=22,color='#222222',label='World net',zorder=3)
ax.set_yticks(y,[f'{r.case_id}  {n}' for (_,r),n in zip(s.iterrows(),names)],fontsize=9)
ax.invert_yaxis();ax.axvline(0,color='#333333',linewidth=.7)
ax.set_xlabel('Intermediate-sourcing contribution, P3 (Mt fossil CO₂)')
ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(FIG/'emitter_contributions.pdf');fig.savefig(FIG/'emitter_contributions.png',dpi=160)
print('Generated two tables and emitter-location figure for ten validated cases.')
