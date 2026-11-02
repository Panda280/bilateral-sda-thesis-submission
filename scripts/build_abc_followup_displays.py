"""Render manuscript displays directly from validated follow-up products."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
OUT=BASE/'followup';TEX=ROOT/'src/chapters/results_revised/generated'
FIG=ROOT/'src/figures/results/hypotheses_20260909'
def main():
    a=pd.read_csv(OUT/'geographic_concentration.csv',dtype={'resolution':str})
    a=a[(a.year==2003)&(a.effect=='D_T_coef')&(a.role=='driver_country')&(a.scope=='Red plus USA')]
    d=pd.read_csv(OUT/'annual_country_attribution.csv',dtype={'resolution':str})
    d=d[(d.year_current==2003)&(d.effect_id=='D_T_coef')&(d.role=='driver_country')]
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{The proposed geographic concentration of the 2003 input-composition effect. Contributions are attributed to structural-driver countries. Negative shares use the sum of negative country nets, not gross detailed-cell effects (R-2026-046).}',r'\label{tab:2003-geography}',r'\begin{tabular}{lrrr}',r'\toprule',r'Target grid & Reduction + USA (Mt) & Share of negative nets & USA (Mt) \\',r'\midrule']
    for row in a.itertuples():
      us=d[(d.resolution==row.resolution)&(d.canonical_country=='USA')].Mt.iloc[0]
      lines.append(f'{row.resolution} & {row.net_Mt:+.1f} & {row.share_negative_country_net_mass*100:.1f}\\% & {us:+.1f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (TEX/'tab_2003_geography.tex').write_text('\n'.join(lines)+'\n')
    gp=pd.read_csv(OUT/'group_period_sector_changes.csv')
    origins=pd.read_csv(OUT/'annual_sector_import_origins.csv')
    red=set(d[d.group.eq('Red')&d.resolution.eq('R3')].label)
    origins=origins[origins.country.isin(red)]
    rows=[]
    for period,sector,name,years in [('P1',25,'Textiles and apparel',range(2001,2005)),('P2',43,'Iron and steel',range(2005,2008)),('P3',74,'Other land transport',range(2008,2013)),('P3',78,'Air transport',range(2008,2013))]:
      r=gp[(gp.group=='Red')&(gp.period==period)&(gp.sector==sector)].iloc[0]
      oo=origins[(origins.year.isin(years))&(origins.sector==sector)].groupby('source_group').delta_imported_Mt.sum()
      assert np.isclose(oo.sum(),r.delta_imported_footprint_Mt)
      rows.append(dict(period=period,sector=sector,label=name,territorial_Mt=r.delta_territorial_Mt,imported_Mt=r.delta_imported_footprint_Mt,non_participant_Mt=oo['Non-Participant'],non_reduction_Mt=oo['Non-Red'],reduction_foreign_Mt=oo['Red']))
    pd.DataFrame(rows).to_csv(OUT/'selected_sector_period_examples.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{Sector patterns hidden by Reduction-group totals, R3 (Mt CO$_2$). The imported footprint is classified by the foreign emitting sector; its Non-Participation component identifies cross-group origins. Examples illustrate all three periods and are not an exhaustive inventory (R-2026-047).}',r'\label{tab:fine-displacement}',r'\begin{tabular}{llrrr}',r'\toprule',r'Period & Emitting sector & Territorial & Imported & Non-Part. origin \\',r'\midrule']
    for r in rows:lines.append(f"{r['period']} & {r['label']} & {r['territorial_Mt']:+.1f} & {r['imported_Mt']:+.1f} & {r['non_participant_Mt']:+.1f} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (TEX/'tab_fine_displacement.tex').write_text('\n'.join(lines)+'\n')
    c=pd.read_csv(OUT/'canada_cereal_cattle_path.csv')
    previous=c[(c.year==2000)&(c.price_basis=='CYP')].iloc[0];current=c[(c.year==2001)&(c.price_basis=='PYP')].iloc[0]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,4.5),gridspec_kw={'width_ratios':[1.1,1]})
    pos=np.arange(2);width=.33
    bars=ax[0].bar(pos-width/2,[previous.CA,previous.US+previous.Other],width,color='#a8b9c8',label='2000 CYP')
    ax[0].bar_label(bars,fmt='%.1f',padding=3)
    bars=ax[0].bar(pos+width/2,[current.CA,current.US+current.Other],width,color='#287e8d',label='2001 PYP')
    ax[0].bar_label(bars,fmt='%.1f',padding=3)
    ax[0].set_xticks(pos,['Canadian inputs','Imported inputs']);ax[0].set_ylabel('Million EUR, 2000 prices');ax[0].set_ylim(0,870);ax[0].legend(frameon=False)
    ax[0].set_title('Cereal inputs into Canadian cattle/milk',loc='left',fontweight='bold')
    values=[current.US-previous.US,current.Other-previous.Other]
    bars=ax[1].barh(['USA (Non-Reduction)','Other foreign suppliers'],values,color=['#b45b36','#a8b9c8'])
    ax[1].bar_label(bars,fmt='+%.1f',padding=3);ax[1].invert_yaxis();ax[1].set_xlim(0,38)
    ax[1].set_xlabel('Import increase, million EUR in 2000 prices');ax[1].set_title('A named supplier across groups',loc='left',fontweight='bold')
    fig.suptitle('Canada: domestic inputs fall while US inputs rise',fontsize=14,x=.02,ha='left')
    fig.text(.02,.915,'R3 cereal group to cattle/raw milk | Canada: Reduction; USA: Non-Reduction | Import share: 8.3% to 13.7%',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(FIG/'canada_cross_group_input.pdf');fig.savefig(FIG/'canada_cross_group_input.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
