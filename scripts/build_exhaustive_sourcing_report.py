"""Results-ready account of the exhaustive retained-data A-D screen."""
from pathlib import Path
import json, html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
B=AN/'exhaustive_sourcing_20260910';FIG=B/'figures';FIG.mkdir(exist_ok=True)
h=pd.read_csv(B/'all_hierarchy_annual.csv',dtype={'grid':str,'sector':str,'parent':str})
c=pd.read_csv(B/'repeated_ABCD_candidates.csv',dtype={'grid':str,'sector':str,'parent':str})
p=pd.read_csv(B/'all_pairs_period.csv',dtype={'source':str,'user':str})
sen=pd.read_csv(B/'threshold_sensitivity.csv')
tr=pd.read_csv(AN/'full_drilldown_20260910/complete_three_grid_periods.csv',dtype={'parent':str})
sm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv',dtype=str)
names=sm.drop_duplicates('fine_sector').set_index('fine_sector').fine_sector_label.to_dict()
cn={'GB':'United Kingdom','CZ':'Czechia','CA':'Canada','DE':'Germany','SK':'Slovakia','DK':'Denmark'}
short={'34':'Basic plastics','35':'Fertilisers','28':'Pulp','94':'Landfill services','43':'Steel','41':'Cement'}
anchor=c.sort_values('grid',ascending=False).drop_duplicates(['effect','country','sector','period']).sort_values('ranking_scale_Mt',ascending=False).copy()
anchor['case_id']=[f'E{i+1:02d}' for i in range(len(anchor))]
anchor.to_csv(B/'report_cases.csv',index=False)
pages=[]
def para(t):return dict(type='paragraph',text=t)
def table(headers,rows,widths):return dict(type='table',headers=headers,rows=rows,widths=widths)
def picture(path,height=200):return dict(type='image',path=str(path),height=height)
def page(title,items):pages.append(dict(title=title,items=items))
def num(x,n=2):return f'{x:+.{n}f}'
def label(s):return names.get(s,s)

page('Exhaustive sourcing and intensity screen',[
 para('All 98 R3 products | All 9,604 supplier-user pairs | All 22 Reduction countries | P1: 2001-04, P2: 2005-07, P3: 2008-12'),
 para('<b>Main finding.</b> Six country-product-period cases satisfy a repeated annual sourcing, intensity and composition test. UK plastics in P3 is the strongest retained accounting illustration: chemical-parent intensity falls while the within-activity contribution is positive. A negative composition contribution more than offsets that increase.'),
 table(['Requested evidence','UK plastics / chemicals, P3'],[
 ['A. Domestic sourcing decreases','Domestic D_C contribution: -2.54 Mt CO2-eq.'],
 ['B. Foreign sourcing increases','Foreign D_C: +4.44 Mt; +2.13 Mt originates outside Reduction.'],
 ['C. Domestic intensity decreases','Domestic chemical-parent chained intensity index falls 32.79%.'],
 ['D. High-intensity product contracts','A-D pass in 2008, 2010, 2011 and 2012. Composition contributes -37.76 index percentage points; within-activity change contributes +4.97.']],[.35,.65]),
 para('The product-specific centered composition allocation is -23.97 percentage points for plastics. It is part of the parent composition total, not an additional effect. The associated parent D_int is approximately -2.22 Mt under either coarse mapping, whereas the R3 effects summed to the same parent are +0.45 Mt.'),
 para('<b>Interpretive limit.</b> A and B are reported sourcing-effect contributions, not observed import quantities. The evidence supports a sourcing/composition association consistent with the proposed mechanism; it does not identify foreign sourcing as the cause of domestic contraction.'),
 para('<b>Indicator boundary.</b> The saved campaign is GWP100 (R-2026-053), not the approved fossil-CO2 result set. This report is conditional exploratory evidence. The reviewed introduction and thesis scope are unchanged.')])

page('Coverage and tests: no product shortlist',[
 table(['Layer','Universe examined','What it answers'],[
 ['Directed D_C links','9,604 pairs x 12 years = 115,248 observations','Which input-to-user sourcing effects have opposing signs?'],
 ['Country-product sourcing','98 products x 22 countries x 12 years x 2 effects = 51,744 observations','Where do D_C or D_YF show national sourcing shifts?'],
 ['Parent intensity/composition','44,510 available annual mapping/effect rows; 58 products have comparable observations','Does the product lose domestic output/share while its parent intensity falls?']],[.25,.37,.38]),
 para('The initial A/B screen includes every retained product and supplier-user pair. Hierarchical C/D tests have additional data requirements: reviewed unique mappings, positive output, a parent that combines at least two fine products in one country, and reproduction of actual parent output and weighted intensity within 1e-5. Other products remain in the A/B inventory; unavailable C/D is not a failed economic hypothesis.'),
 para('<b>Annual A-D rule.</b> A: domestic contribution below -0.1 Mt. B: foreign contribution above +0.1 Mt. C: domestic parent coefficient v1 is below v0. D: the product starts above parent intensity, loses absolute output and output share, and has a negative centered composition contribution within a parent with negative total composition. All conditions must coincide in a year.'),
 para('<b>Period rule.</b> At least two annual passes, complete comparable period coverage, matching period sourcing signs above 0.1 Mt, and negative chained parent intensity and composition. This is stricter than the earlier 11-case screen: actual product output must decline and the parent decline must persist at period level. Foreign Red is now included and reported separately.'),
 para('C is a coefficient change; D_int is an emission-effect contribution. They are different quantities. Neither the sourcing-only nor A-D selection is a statistical significance test. Candidate counts depend on the stated materiality and persistence rules.')])

page('Six repeated A-D cases',[
 para('Domestic and foreign columns are country-product D_C contributions in Mt CO2-eq. Foreign includes other Reduction countries. The last columns are changes in a normalized domestic parent intensity index, not emission effects.'),
 table(['Case','Period / years','Domestic','Foreign','Outside Red','Intensity %','Composition pp','Within pp'],[
 [f'{r.country} {short[r.sector]}',f'{r.period}: {r.annual_ABCD}',num(r.domestic_Mt),num(r.foreign_Mt),num(r.outside_Mt),num(r.chained_intensity_change_pct),num(r.chained_composition_pp),num(r.chained_within_pp)] for r in anchor.itertuples()],[.19,.10,.11,.11,.11,.12,.13,.13]),
 para('UK plastics, Czech fertilisers and Danish cement pass under both coarse mappings. Canadian pulp and German landfill pass only under the 2016-compatible mapping; Slovak steel passes only under 2013. This refers to qualification, not independent database replication.'),
 para('<b>Composition necessary in the diagnostic:</b> UK chemicals, German waste services and Danish non-metallic minerals have positive period within-activity contributions but negative total intensity changes. Their composition contributions account for the reversal. Landfill remains a service case, with no inference of physical waste exports.'),
 para('Danish cement is a new example beyond the previous highlighted list. Its foreign sourcing increase is mainly within Reduction: +0.236 Mt from other Reduction countries versus +0.065 Mt from outside. This distinguishes national sourcing substitution from a mechanism that displaces emissions outside the group.'),
 para('All six involve D_C. No D_YF case survives the joint period test at 0.1 Mt. The complete unfiltered annual and period inventories are retained alongside this report.')])

tops=p[p.repeated_foreign].sort_values('ranking_scale_Mt',ascending=False)
for start in [0,15]:
    z=tops.iloc[start:start+15]
    page(f'All-pairs discovery: ranks {start+1}-{start+len(z)}',[
     para('Ranked among 379 repeated A/B pair-periods by min(-domestic, foreign) for ordering only. This ranking scale is not an offshoring estimate. All emitter countries and individual using countries are summed; these are Reduction-group links, not identified country-specific supply chains.'),
     table(['Period','Supplier -> using industry (R3)','Domestic','Foreign','Outside Red','Years'],[
      [r.period,f'{r.source} {label(r.source)[:38]} -> {r.user} {label(r.user)[:38]}',num(r.domestic_Mt),num(r.foreign_Mt),num(r.outside_Mt),str(r.annual_AB_foreign)] for r in z.itertuples()],[.09,.44,.12,.12,.13,.10]),
     para('These are A/B candidates only. Sea transport, crude-petroleum supply, petroleum refining and steel inputs to fabricated metals emerge without prespecifying those links. A large sourcing contribution alone does not establish a declining domestic intensity or the proposed composition mechanism.')])

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
for r in anchor.itertuples():
    q=h[(h.grid==r.grid)&(h.effect==r.effect)&(h.country==r.country)&(h.sector==r.sector)&(h.period==r.period)].sort_values('year')
    fig,ax=plt.subplots(1,2,figsize=(8.3,3.2));x=np.arange(len(q));w=.25
    ax[0].bar(x-w,q.Domestic,w,label='Domestic',color='#245d7c')
    ax[0].bar(x,q.Red,w,label='Foreign Red',color='#779358')
    ax[0].bar(x+w,q.outside_source,w,label='Outside Red',color='#c46a36')
    ax[0].set_ylabel('D_C contribution (Mt CO2-eq)');ax[0].set_title('A / B: sourcing contributions')
    ax[1].bar(x-.15,q.composition_relative*100,.3,label='Composition',color='#245d7c')
    ax[1].bar(x+.15,q.within_relative*100,.3,label='Within activities',color='#c46a36')
    ax[1].plot(x,q.parent_change_pct,'ko-',ms=3,label='Net intensity change')
    ax[1].set_ylabel('% of annual initial parent intensity');ax[1].set_title('C / D: parent intensity change')
    for a in ax:
        a.axhline(0,color='#444',lw=.6);a.set_xticks(x,[str(y)+('*' if passed else '') for y,passed in zip(q.year,q.ABCD)])
        a.legend(fontsize=7,loc='best');a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
    fig.tight_layout();path=FIG/f'{r.case_id}.png';fig.savefig(path,dpi=180);plt.close(fig)
    items=[para(f'R3 {r.sector} {html.escape(label(r.sector))}. Anchor parent: {r.grid}-compatible {r.parent}. A-D qualifying years: {r.qualifying_years}. An asterisk marks an annual pass.'),
      picture(path,200),
      table(['Year','Product / parent intensity','Product output %','Output share pp','Parent intensity %','Product comp. %'],[
       [str(z.year),f'{z.product_intensity_ratio:.2f}',num(z.output_change_pct),num(100*z.share_change),num(z.parent_change_pct),num(100*z.centered_product_composition/z.v0)] for z in q.itertuples()],[.10,.20,.17,.16,.19,.18]),
      para(f'<b>Period:</b> domestic D_C {num(r.domestic_Mt)} Mt; foreign {num(r.foreign_Mt)} Mt, including {num(r.outside_Mt)} Mt outside Reduction. Parent chained intensity {num(r.chained_intensity_change_pct)}%; composition {num(r.chained_composition_pp)} pp and within-activity {num(r.chained_within_pp)} pp. The selected product accounts for {num(r.chained_selected_product_composition_pp)} pp of centered composition.'),
      para(f'Period parent D_int: {num(r.parent_Dint_Mt)} Mt; fine effects summed within the anchor parent: {num(r.fine_parent_Dint_Mt)} Mt. These are separate from the coefficient diagnostics. Period domestic/foreign sourcing signs '+('survive every single-year omission.' if r.source_sign_stable_omit_year else 'do not survive every single-year omission.'))]
    if r.sector=='94':items.append(para('Landfill sourcing is a service attribution. The result does not establish physical waste shipments.'))
    elif r.country=='DK':items.append(para('Most of the foreign increase is within Reduction. In 2007 the domestic sourcing contribution reverses and parent intensity rises; the complete P2 index nevertheless declines.'))
    elif r.country=='GB':items.append(para('In 2011, plastics starts at 1.64 times parent intensity and loses 19.02 percentage points of output share. Parent composition is -25.80% and within-activity change +6.44%, giving a net -19.36%. This is the clearest single-year illustration of D.'))
    page(f'{r.case_id} | {cn[r.country]}: {short[r.sector]} ({r.period})',items)

page('Hierarchy, persistence and sensitivity',[
 para('The fine-to-coarse comparison uses EXIOBASE-derived compatible grids, not three independent databases. Coefficients are output-weighted averages; calculated SDA effects are summed without another set of output weights. Recalculating the SDA at a coarser resolution can change its attribution.'),
 table(['Same 2013 parent','R3 effects','2016 effects','2013 effect'],[
  [f'{country} {par} ({per})',num(z.R3_Mt),num(z.WIOT2016_Mt),num(z.WIOT2013_Mt)]
  for country,par,per in [('GB','24','P3'),('CZ','24','P2'),('DK','26','P2')]
  for z in tr[(tr.country==country)&(tr.parent==par)&(tr.period==per)&tr.complete].itertuples()],[.4,.2,.2,.2]),
 para('All values above are Mt CO2-eq. UK chemicals and Danish non-metallic minerals both exhibit fine/coarse sign reversals. Czech chemicals retains a negative sign across the three grids. The A-D screen does not require a sign reversal.'),
 table(['Annual / period floor (Mt)','All foreign: distinct A-D cases','Outside Red: distinct A-D cases'],[
  [str(t),str(sen[(sen.threshold_Mt==t)&(sen.foreign_definition=='foreign')].distinct_cases.iloc[0]),str(sen[(sen.threshold_Mt==t)&(sen.foreign_definition=='outside_source')].distinct_cases.iloc[0])] for t in sorted(sen.threshold_Mt.unique())],[.32,.34,.34]),
 para('The sign-only test includes small effects. At 0.1 Mt there are six national cases but only four outside-Reduction cases. No case retains two qualifying years at 1 Mt. These thresholds measure materiality, not statistical significance.'),
 para('Large annual cases are not automatically stable period cases. German coke in 2005 combines a very large domestic contribution with a 95.75% fall in recorded product output. It remains in the inventory but requires particular input-data scrutiny; it is not presented as a clear economic relocation episode. UK coke and German fish also demonstrate why annual passes alone are insufficient.')])

page('Interpretation and reproducibility',[
 para('The exhaustive retained-data search supports a bounded conclusion: some national sourcing shifts coincide with lower domestic parent intensity and contraction of an above-parent-intensity product. UK plastics provides the strongest recurring accounting illustration. Danish cement provides an additional example dominated by trade within Reduction. These patterns are consistent with compositional intensity improvement; they do not quantify the causal contribution of offshoring to group target achievement.'),
 para('<b>Coefficient diagnostic.</b> For each annual transition, v(parent) = sum[w(i)v(i)]. The midpoint identity splits its change into sum[mean(w) change(v)] (within) and sum[mean(v) change(w)] (composition). Centered product composition replaces mean(v(i)) with mean(v(i))-mean(v(parent)); these centered terms still sum to total composition because output-share changes sum to zero.'),
 para('Period indices begin at 1 and multiply annual v1/v0 ratios, each using CYP(t-1) and PYP(t). Annual within and composition changes divided by v0 are multiplied by the previous index and accumulated. Their sum exactly reconstructs the final index change. This is a chained coefficient diagnostic, not a new SDA subeffect or a fixed-price endpoint comparison.'),
 para('<b>What remains unidentified:</b> actual changes in physical import quantities; the country-specific supplier-user pathway behind a group link; whether foreign sourcing caused domestic contraction; low production costs; and the portion of legal target achievement attributable to offshoring. The saved GWP100 indicator and unresolved input-provenance issue must be resolved before these become approved fossil-CO2 thesis findings.'),
 para('<b>Retained evidence.</b> The package contains all pair-year/period observations, all country-product observations, all available hierarchy tests including failures, all six selected cases and threshold sensitivity. Pair and country-product D_C margins reconcile within 6.94e-6 scaled error, below the retained FP32 tolerance of 1e-5.'),
 para('Reproduce with exhaustive_sourcing_intensity.py, then build_exhaustive_sourcing_report.py; render report_content.json with render_drilldown_pdf.py using this package as --content-dir. Results are registered as R-2026-057. The source hashes are in validation.json; delivery checks are in delivery_validation.json.')])

(B/'report_content.json').write_text(json.dumps(pages,indent=2),encoding='utf-8')
md=[]
for pg in pages:
    md += ['# '+pg['title'],'']
    for it in pg['items']:
        if it['type']=='paragraph':md += [it['text'].replace('<b>','**').replace('</b>','**'),'']
        elif it['type']=='table':
            md += ['| '+' | '.join(it['headers'])+' |','| '+' | '.join(['---']*len(it['headers']))+' |']
            md += ['| '+' | '.join(map(str,row))+' |' for row in it['rows']];md += ['']
        else:md += ['![]('+str(Path(it['path']).relative_to(B)).replace('\\','/')+')','']
(B/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
print(f'{len(pages)} prepared pages; {len(anchor)} distinct case pages')
