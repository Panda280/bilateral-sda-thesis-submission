"""Create illustrated report content from retained, validated drill-down tables.

Run with project Python for charts/content; render_drilldown_pdf.py uses the
bundled reportlab runtime. No source effect is recalculated here.
"""
from pathlib import Path
import json,html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];AN=ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
B=AN/'full_drilldown_20260910';FIG=B/'figures';FIG.mkdir(exist_ok=True)
c=pd.read_csv(B/'highlighted_cases.csv',dtype={'grid':str,'sector':str,'parent':str})
j=pd.read_csv(B/'all_annual_joint_tests.csv',dtype={'grid':str,'sector':str,'parent':str})
tr=pd.read_csv(B/'complete_three_grid_periods.csv',dtype={'parent':str})
links=pd.read_csv(B/'directed_supply_links_period.csv',dtype={'source':str,'using_sector':str})
context=pd.read_csv(B/'directed_link_coarse_context.csv',dtype={'grid':str})
sm=pd.read_csv(ROOT/'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv',dtype=str)
allfine=pd.read_csv(AN/'hierarchical_intensity_20260910/fine_cell_diagnostics.csv',dtype={'grid':str,'sector':str,'parent':str})
sector_names=sm.drop_duplicates('fine_sector').set_index('fine_sector').fine_sector_label.to_dict()
component_exports=[]
countries={'CA':'Canada','DE':'Germany','GB':'United Kingdom','CZ':'Czechia','JP':'Japan'}
products={'28':'Pulp and secondary-pulp reprocessing','43':'Steel and ferro-alloys','94':'Landfill of food and other waste','35':'Fertilisers','25':'Textiles and wearing apparel','23':'Fish products','31':'Coke oven products','34':'Basic plastics and reprocessing'}
short={'28':'pulp','43':'steel','94':'landfill','35':'fertilisers','25':'textiles','23':'fish products','31':'coke','34':'basic plastics'}
pages=[]
def para(text):return {'type':'paragraph','text':text}
def table(headers,rows,widths=None):return {'type':'table','headers':headers,'rows':rows,'widths':widths}
def picture(name,height):return {'type':'image','path':str(FIG/name),'height':height}
def page(title,items):pages.append({'title':title,'items':items})
def num(x,n=2):return f'{x:+.{n}f}'
def mapping(sector,grid):
 z=sm[(sm.fine_sector==sector)&(sm.coarse_resolution==grid)].iloc[0]
 return z.coarse_sector if z.mapping_status=='reviewed_unique' else z.candidate_coarse_sectors

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
fig,ax=plt.subplots(figsize=(8.2,3.7));y=np.arange(len(c))
ax.barh(y-.16,c.home_Mt,.3,label='Home emitters',color='#22638a')
ax.barh(y+.16,c.outside_emitter_Mt,.3,label='Emitters outside Reduction',color='#bc632d')
ax.set_yticks(y,[f'{r.case_id}  {r.country} {short[r.sector]} ({r.period})' for r in c.itertuples()]);ax.invert_yaxis()
ax.axvline(0,color='#333333',lw=.7);ax.set_xlabel('Selected source-product D_C contribution (Mt CO2-eq)')
ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(FIG/'case_overview.png',dpi=180);plt.close(fig)

page('Full drill-down: sourcing and intensity',[
 para('R3 -> WIOD2016-compatible -> WIOD2013-compatible grids | P1 2001-04, P2 2005-07, P3 2008-12 | 10 September 2026'),
 para('<b>Indicator boundary.</b> These are the stored EXIOBASE GWP100 results (CO2-equivalent), confirmed by R-2026-053. They are conditional findings for that campaign and cannot be presented as fossil-CO2 results. The reviewed introduction and selected thesis scope are unchanged.'),
 para('The full screen finds 11 distinct country-product-period candidates with at least two qualifying annual transitions: three in P1, four in P2 and four in P3. Five repeat under both coarse mappings. All are intermediate-sourcing effects; no final-demand-sourcing candidate repeats under the same screen.'),
 table(['Period','Distinct cases','Under both mappings'],[['P1','3','0'],['P2','4','1'],['P3','4','4']],[.34,.33,.33]),
 picture('case_overview.png',250),
 para('The bars are selected contributions, not estimates of total offshoring. Two cases concern landfill services, and no physical waste exports are established. Four cases retain the required period sourcing signs after every single-year omission; none repeats at the stricter 1 Mt annual threshold.')])

page('What the hierarchy can reveal',[
 para('The hypothesis is that a high-intensity fine product loses domestic output share while sourcing shifts towards other groups. The coarse domestic sector can then become less intensive through its composition, even when its remaining fine activities do not improve.'),
 para('For positive output, the coarse coefficient is an output-weighted average: v(parent) = sum[w(i) v(i)], where w(i) = x(i)/sum[x]. Its exact annual change can be written as sum[mean(w) change(v)] + sum[mean(v) change(w)]. The first term measures within-activity change; the second measures composition. These are coefficient diagnostics, not additional SDA effects.'),
 para('Calculated fine D_int contributions are summed without weights. A separate coarse recalculation need not reproduce that sum. The two operations answer different questions: what detail is hidden by reporting, and how aggregation changes the calculated attribution.'),
 table(['R3 example','2016-compatible parent','2013-compatible parent'],[
 ['Pulp (28)','C17 paper products','21t22 paper/printing'],['Steel (43)','C24 basic metals','27t28 basic/fabricated metals'],
 ['Coke (31)','C19 coke/petroleum','23 coke/petroleum'],['Plastics (34), fertilisers (35)','C20 chemicals','24 chemicals'],
 ['Crops (1-3), livestock (4-6)','A01 agriculture','AtB agriculture/forestry/fishing'],
 ['Textiles (25)','C13-C15 textiles/leather','17t18 textiles: chain is not nested'],['Landfill (94)','E37-E39 waste services','E utilities (reviewed mapping)']],[.33,.34,.33]),
 para('The 2016-compatible C13-C15 cell spans two 2013 cells (17t18 and 19), so no deterministic chain is imposed. R3 composites 54, 86 and 96 also prevent some exact mappings. The stale 48-sector source concordance is not used to override the reviewed active-grid mappings.'),
 para('Coverage: 8,969 of 10,489 screened direct R3 parent-years reproduce parent output/intensity; 12,659 of 14,352 screened 2016-to-2013 parent-years do so. A fully audited three-level chain exists for 3,915 parent-years, yielding 944 complete parent-period comparisons. These are coverage counts, not representative population shares.')])

page('Selection, persistence and accounting boundaries',[
 para('Each annual candidate must satisfy every gate: initial product intensity exceeds the domestic parent average; its domestic output share falls; parent intensity falls; the parent composition term is negative; and the coarse parent D_int is negative. Domestic-source and home-emitter D_C contributions must each be below -0.1 Mt CO2-eq, while outside-Reduction source and emitter contributions must each exceed +0.1 Mt.'),
 para('A repeated candidate passes in at least two annual transitions. The same tests are applied separately to D_C and D_YF. All 11 highlighted cases have complete period coverage and opposing period sourcing signs. Raising the annual materiality threshold to 1 Mt leaves no candidate with two qualifying years. Materiality thresholds are not statistical significance tests.'),
 para('A source-product case covers sourcing by one country across its using industries. Its home/foreign emitter partition covers all emitting industries. Source-group and emitter-location partitions describe the same effect and are not added. In particular, foreign Red is distinct from domestic and outside Reduction.'),
 para('The directed supply-link analysis later in this report does retain both source product and using sector, but aggregates purchasing countries to Reduction and sums emitters. It can identify coke-to-steel or crop-to-livestock sourcing contributions at group level; it does not identify a particular country-level bilateral link.'),
 para('Every period is a sum of adjacent-year CYP(t-1) -> PYP(t) effects. A mean annual composition percentage is not an endpoint intensity change. Pure-sector comparisons exclude country merges, nonpositive fine outputs and failed parent reproduction checks. Recalculated effects may still respond to aggregation elsewhere in the production network.'),
 para('Reproducibility: source-product/emitter slices are unique, and their country marginals reconcile to registered effects within a maximum scaled discrepancy of 7.431e-6, inside the existing FP32 tolerance of 1e-5. The extraction does not alter matrices, mappings or saved effects. Results are registered as R-2026-055 and R-2026-056.')])

rows=[]
for r in c.itertuples():
 rows.append([r.case_id,r.period,f'{r.country}: {short[r.sector]}',r.qualifying_years,'Both' if r.both_mappings else r.grid,
              num(r.home_Mt),num(r.outside_emitter_Mt),num(r.parent_Dint_Mt),'Yes' if r.source_sign_stable_omit_year else 'No'])
page('All 11 repeated candidates',[
 para('Mt CO2-equivalent. Parent D_int uses the highlighted mapping (2016 where it qualifies, otherwise 2013). Parent boundaries differ by row; these parent effects must not be summed into a group total. The last column asks whether all four period sourcing signs survive removal of any one year.'),
 table(['ID','Period','Country: product','Passing years','Mapping','Home','Outside Red','Parent D_int','Stable'],rows,[.06,.055,.17,.16,.075,.085,.10,.105,.09]),
 para('<b>Full-period distinction.</b> H08 German fish products and H09 UK coke pass in two P3 years, but their parent sectors have positive D_int over P3. They are repeated annual episodes, not persistent period-wide parent intensity reductions. H03 UK landfill retains opposing period signs but its home/domestic net is below the 0.1 Mt period materiality floor.'),
 para('<b>More robust sourcing signs.</b> H05 German landfill, H06 Japanese textiles, H10 UK plastics and H11 Japanese coke retain all required period sourcing signs when any one year is removed. This does not establish causality or robustness of every composition diagnostic.')])

strong=tr[tr.complete&tr.group.eq('Red')].copy()
strong['gap']=strong.WIOT2013_Mt-strong.R3_Mt
strong['flip']=(strong.R3_Mt*strong.WIOT2013_Mt<0)&(abs(strong.R3_Mt)>.1)&(abs(strong.WIOT2013_Mt)>.1)
top=strong[strong.flip].assign(magnitude=lambda x:abs(x.gap)).sort_values('magnitude',ascending=False).head(8)
top.to_csv(B/'largest_material_grid_reversals.csv',index=False)
page('Where aggregation changes the sign',[
 para('Each row below uses the same 2013 parent boundary: sum its R3 effects, sum its 2016 effects, then read its independently recalculated 2013 effect. Only complete-period, fully audited chains are admitted. Both endpoint magnitudes must exceed 0.1 Mt for this table.'),
 table(['Country','Period','2013 parent','R3 sum','2016 sum','2013 effect'],[[r.country,r.period,r.parent,num(r.R3_Mt),num(r.WIOT2016_Mt),num(r.WIOT2013_Mt)] for r in top.itertuples()],[.14,.12,.20,.18,.18,.18]),
 para('This is a sensitivity table, not a further list of offshoring findings: a sign reversal alone does not pass the joint sourcing/composition screen.'),
 para('<b>UK chemicals, P3.</b> The complete-period D_int changes from +0.45 Mt at R3 to -2.22 Mt on each coarse grid. In 2011 specifically, the output-composition term is -25.80% of initial parent intensity while the within-activity term is +6.44%. The parent becomes less intensive even though the within term increases intensity. The fine parent effects sum to +0.409 Mt in that year, versus -1.114 Mt on the 2016 grid.'),
 para('<b>Czech chemicals, P2.</b> All three contributions are negative, but their magnitudes differ: -2.72 Mt at R3, -5.23 Mt on the 2016 grid and -5.24 Mt on the 2013 grid. This is amplification of a reduction, not a sign reversal.'),
 para('Grid differences are not separately identified offshoring effects. The parent-level composition identity and the reported sourcing contributions supply additional evidence, but neither attributes the complete grid gap causally to foreign sourcing.')])

page('Directed links: coke, crops and livestock',[
 para('These are D_C interactions into Reduction using industries, with all emitters summed. The domestic-source and outside-Reduction columns are parts of one sourcing effect. Outside combines Non-Reduction and Non-Participation; foreign Red is omitted from this table but retained in the data.'),
 table(['Link','Period','Domestic','Outside Red','Opposing years'],[[r.link,r.period,num(r.domestic_Mt),num(r.outside_Mt),f'{r.opposing_years}/{r.years}'] for r in links.itertuples()],[.43,.10,.15,.16,.16]),
 para('Three links combine opposing period contributions with at least two annual passes: coke-to-steel in P1 (-2.70/+2.83 Mt), cereals-to-cattle/milk in P1 (-0.24/+0.40 Mt), and other crops-to-cattle/milk in P2 (-1.16/+1.04 Mt). These match the direction of the proposed upstream sourcing mechanism.'),
 para('None of the comparable agricultural country-product cells passes the full intensity/composition/sourcing screen. That does not contradict the directed crop links: the joint screen imposes additional country-level and intensity conditions. Physical feed use, trade volumes and production relocation are not observed by these effects.')])

chosen=[('Coke to steel','P1'),('Cereals to cattle/milk','P1'),('Other crops to cattle/milk','P2')]
ctx=pd.concat([context[(context.link==label)&(context.period==period)] for label,period in chosen])
page('How the directed links become hidden',[
 para('Following a fine link into its coarse parent broadens the source and using activities. The rows below deliberately show that broadening; their values are not three estimates of the same fine link. Country coverage also differs between grids.'),
 table(['Fine-link context','Grid','Source -> using parent','Domestic','Outside Red'],[[r.link+' ('+r.period+')',r.grid,r.source_parent+' -> '+r.using_parent,num(r.domestic_Mt),num(r.outside_Mt)] for r in ctx.itertuples()],[.31,.10,.27,.16,.16]),
 para('Coke (31) and steel (43) remain distinct parents at both coarse levels: C19 -> C24, then 23 -> 27t28. The 2013 using parent additionally includes fabricated metals. A negative domestic coke-sourcing contribution therefore cannot be interpreted as a within-steel-sector direct-intensity change.'),
 para('Crop and livestock links collapse into A01 -> A01 on the 2016 grid. The 2013 AtB -> AtB cell also includes forestry and fishing. A coarse same-sector interaction consequently conceals both the identity of the supplied crop and the livestock activity using it.'),
 para('This distinction is central to the thesis: aggregation can hide a specific supply link even when its overall parent contribution remains visible. It can also change the calculated contribution after the system is rebuilt. The two effects should be reported separately.')])

for r in c.itertuples():
 a=j[(j.grid==r.grid)&(j.period==r.period)&(j.country==r.country)&(j.sector==r.sector)&(j.effect==r.effect)].sort_values('year')
 assert len(a)==r.years and int(a.signature.sum())==r.passes
 p16=mapping(r.sector,'2016');p13=mapping(r.sector,'2013')
 fig,axes=plt.subplots(1,2,figsize=(8.2,2.65));xx=np.arange(len(a));w=.34
 ax=axes[0];ax.bar(xx-w/2,a.Domestic/1e9,w,label='Domestic source',color='#22638a');ax.bar(xx+w/2,a.outside_source/1e9,w,label='Outside-Red source',color='#bc632d')
 ax.set_title('Source-product D_C');ax.set_ylabel('Mt CO2-eq')
 ax=axes[1];ax.bar(xx-w/2,a.fine_Dint_Mt,w,label='R3 sum in parent',color='#66858e');ax.bar(xx+w/2,a.coarse_Dint_Mt,w,label=f'{r.grid} parent',color='#624178')
 ax.set_title(f'Parent intensity effect ({r.parent})')
 for ax in axes:
  ax.axhline(0,color='#333333',lw=.7);ax.set_xticks(xx,[str(y)+('*' if yes else '') for y,yes in zip(a.year,a.signature)])
  ax.legend(fontsize=7,loc='best');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 fig.tight_layout();fig.savefig(FIG/f'{r.case_id}.png',dpi=180);plt.close(fig)
 matched=tr[(tr.country==r.country)&(tr.period==r.period)&(tr.parent==p13)&tr.complete]
 if len(matched)==1:
  z=matched.iloc[0];tb=table(['Common 2013 parent boundary','R3 sum','2016 sum','2013 effect'],[[p13,num(z.R3_Mt),num(z.WIOT2016_Mt),num(z.WIOT2013_Mt)]],[.43,.19,.19,.19])
 else:
  tb=table([f'Anchor parent ({r.grid}: {r.parent})','R3 sum','Coarse effect'],[['Period D_int (Mt CO2-eq)',num(r.fine_parent_Dint_Mt),num(r.parent_Dint_Mt)]],[.55,.225,.225])
 state=('The parent intensity effect remains negative over the full period.' if r.parent_Dint_Mt<0 else 'The parent intensity effect is positive over the full period: the annual passes do not establish a period-wide parent reduction.')
 stability=('Period sourcing signs survive every single-year omission.' if r.source_sign_stable_omit_year else 'At least one single-year omission reverses a required period sourcing sign.')
 rownotes='The annual table reports the selected product relative to its domestic anchor parent. Composition is the full parent composition term, not the selected product alone.'
 special=''
 if r.sector=='94':special='Landfill is a treatment-service group. These effects do not establish physical waste exports.'
 elif r.sector=='25':special='C13-C15 is not nested in a single 2013 parent; no full three-step inference is made.'
 elif r.sector=='31':special='This is composition within coke/petroleum manufacturing. It does not identify coke use in steel production in this country.'
 elif not len(matched):special='The complete three-level parent comparison is not admitted; the direct anchor comparison is retained.'
 annualrows=[[str(v.year)+('*' if v.signature else ''),f'{v.initial_v/v.parent_v:.2f}',num(v.share_change*100),num(v.composition_relative*100),num(v.within_relative*100)] for v in a.itertuples()]
 components=allfine[(allfine.grid==r.grid)&(allfine.period==r.period)&(allfine.country==r.country)&(allfine.parent==r.parent)&allfine.comparable].groupby('sector').agg(Dint_Mt=('Dint_Mt','sum'),years=('year','size')).reset_index()
 assert components.years.eq(r.years).all()
 assert abs(components.Dint_Mt.sum()-r.fine_parent_Dint_Mt)<1e-7
 components['label']=components.sector.map(sector_names);components['case_id']=r.case_id;component_exports.append(components)
 largest=components.assign(magnitude=abs(components.Dint_Mt)).sort_values('magnitude',ascending=False).head(3)
 if r.sector not in set(largest.sector):largest=pd.concat([largest,components[components.sector==r.sector]])
 componentrows=[[v.sector,('Selected product: ' if v.sector==r.sector else '')+v.label[:66]+('...' if len(v.label)>66 else ''),num(v.Dint_Mt)] for v in largest.itertuples()]
 page(f'{r.case_id} | {countries[r.country]}: {products[r.sector]}',[
  para(f'{r.period} | R3 {r.sector} -> 2016 {p16} -> 2013 {p13}. The screen repeats under '+('both coarse mappings.' if r.both_mappings else f'the {r.grid} mapping only.')),
  para(f'Qualifying transitions: {r.qualifying_years}. Period source-product contributions are {num(r.domestic_Mt)} Mt from domestic sources and {num(r.outside_source_Mt)} Mt from outside Reduction. In the emitter partition, the corresponding contributions are {num(r.home_Mt)} Mt at home and {num(r.outside_emitter_Mt)} Mt outside Reduction.'),
  tb,picture(f'{r.case_id}.png',188),
  table(['Year (* passes)','Initial product / parent intensity','Product share change (pp)','Parent composition (% initial v)','Parent within (% initial v)'],annualrows,[.16,.21,.21,.21,.21]),
  table(['R3 code','Largest fine contributors inside anchor parent','Period D_int (Mt)'],componentrows,[.11,.70,.19]),
  para(rownotes+' All fine contributors are retained in case_fine_contributors.csv.'),para(state+' '+stability+' '+special)])

page('Interpretation and reproducibility',[
 para('The evidence supports a bounded statement: the stored GWP100 campaign contains repeated country-product sourcing patterns associated with domestic output composition changes. Some are visible under both coarse mappings; some emerge only at one boundary; some disappear or reverse in period totals.'),
 para('The most informative manufacturing examples are UK plastics (P3), Czech fertilisers (P2), Japanese coke (P3) and German steel (P1). The landfill examples reveal strong attribution sensitivity but do not establish physical trade in waste. Directed coke-to-steel and crop-to-livestock group links are separately observable in the reported D_C interactions.'),
 para('No low production cost is measured. High emissions per euro can reflect physical technology, prices, product mix or the indicator boundary. No causal offshoring quantity, policy effect, target-compliance fraction or export-driven denominator mechanism is identified. Selected candidates are not a group-total estimator.'),
 para('Indicator correction remains essential: the stored campaign matches GWP100 after its intensity policy, whereas the reviewed thesis scope specifies fossil CO2. The conditional findings must be repeated on corrected fossil-CO2 effects before being presented as that thesis result. The independent balanced-versus-consumed-input provenance issue also remains unresolved.'),
 para('Retained package: implementation/bilateral_sda_attribution/docs/analysis/full_drilldown_20260910/. The complete annual screen, all candidates and exclusions, country sourcing reconciliation, directed links, parent comparisons and report figures are stored there. R-2026-055 and R-2026-056 register the numerical results.'),
 para('Reproduce with analyse_full_drilldown.py and drilldown_supply_links.py after the retained P1/P2 and P3 extracts are available. build_full_drilldown_report.py prepares the figures and report content; render_drilldown_pdf.py renders the PDF. Source matrices and saved effects are unchanged. The reviewed introduction is unchanged.')])

(B/'report_content.json').write_text(json.dumps(pages,indent=2),encoding='utf-8')
pd.concat(component_exports,ignore_index=True).to_csv(B/'case_fine_contributors.csv',index=False)
md=['# Full hierarchical drill-down\n']
for pg in pages:
 md.append('## '+pg['title']+'\n')
 for item in pg['items']:
  if item['type']=='paragraph':md.append(item['text'].replace('<b>','**').replace('</b>','**')+'\n')
  elif item['type']=='table':
   md.append('| '+' | '.join(item['headers'])+' |\n| '+' | '.join('---' for _ in item['headers'])+' |')
   md.extend('| '+' | '.join(str(v) for v in row)+' |' for row in item['rows']);md.append('')
  elif item['type']=='image':md.append('!['+pg['title']+']('+Path(item['path']).as_posix()+')\n')
(B/'REPORT.md').write_text('\n'.join(md),encoding='utf-8')
print(f'Prepared {len(pages)} report pages, 11 case figures and complete Markdown report.')
