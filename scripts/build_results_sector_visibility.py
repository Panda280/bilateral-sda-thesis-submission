"""Distinguish cancellation inside a mapped sector from coarse-grid recalculation."""
from pathlib import Path
import hashlib,json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909/sectors'
FIG=ROOT/'src/figures/results/hypotheses_20260909'
def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    pp=ROOT/'data/processed/provisional_panel_v2/cross_year_panel/driver_sector.parquet'
    mp=ROOT/'references/current_codebase/concordance/normalized_2026-08-31/granularity_sector_mapping_review.csv'
    p=pd.read_parquet(pp); p=p[p.year_current.between(2001,2012)].copy()
    assert p.validation_status.eq('passed').all() and p.is_publishable.all() and p.units.eq('kg CO2').all()
    p['period']=p.year_current.map(lambda y:'P1' if y<=2004 else 'P2' if y<=2007 else 'P3')
    p['label']=p.label.astype(str); m=pd.read_csv(mp,dtype=str)
    admitted=m[(m.coarse_resolution=='2016')&(m.mapping_status=='reviewed_unique')].copy()
    assert not admitted.fine_sector.duplicated().any()
    excluded=m[(m.coarse_resolution=='2016')&(m.mapping_status!='reviewed_unique')]
    excluded.to_csv(OUT/'mapping_exclusions.csv',index=False)
    f=p[p.resolution=='R3'].merge(admitted,left_on='label',right_on='fine_sector',validate='many_to_one')
    fine=f.groupby(['period','effect_id','fine_sector','fine_sector_label','coarse_sector'],as_index=False).contribution.sum()
    fine['contribution_Mt']=fine.contribution/1e9
    records=[]
    for key,q in fine.groupby(['period','effect_id','coarse_sector']):
        vals=q.contribution.to_numpy()/1e9; net=vals.sum(); mass=abs(vals).sum()
        period,effect,sector=key
        native=p[(p.resolution=='2016')&(p.period==period)&(p.effect_id==effect)&(p.label==sector)].contribution.sum()/1e9
        records.append({'period':period,'effect':effect,'coarse_sector':sector,'fine_sector_count':len(q),
                        'mapped_R3_net_Mt':net,'positive_fine_nets_Mt':vals[vals>0].sum(),
                        'negative_fine_nets_Mt':vals[vals<0].sum(),'sum_absolute_fine_nets_Mt':mass,
                        'hidden_offset_Mt':(mass-abs(net))/2,'sector_cancellation':1-abs(net)/mass if mass else 0,
                        'recalculated_2016_net_Mt':native,'recalculation_gap_Mt':native-net})
    result=pd.DataFrame(records)
    fine.to_csv(OUT/'fine_sector_period_effects.csv',index=False)
    result.to_csv(OUT/'within_parent_visibility.csv',index=False)
    # Explicit agriculture contrast, specified before selection of numerical examples.
    ag=fine[fine.coarse_sector=='A01'].copy()
    assert set(ag.fine_sector)==set(map(str,range(1,7)))|{'97'}
    fig,axes=plt.subplots(2,3,figsize=(11,6.8),sharex='row')
    labels=['Rice/wheat/cereals','Vegetables/oil seeds','Other crops','Cattle/raw milk','Pigs/poultry','Other animal products','Manure treatment']
    for r,effect in enumerate(['D_C','D_T_coef']):
      for c,period in enumerate(['P1','P2','P3']):
        ax=axes[r,c]; q=ag[(ag.period==period)&(ag.effect_id==effect)].sort_values('fine_sector')
        v=q.contribution_Mt.to_numpy()
        ax.barh(range(7),v,color=np.where(v>=0,'#b45b36','#287e8d'))
        ax.axvline(0,color='#283847',lw=.7); ax.set_yticks(range(7),labels if c==0 else ['']*7);ax.invert_yaxis()
        ax.set_title(period+' | '+('Intermediate sourcing' if r==0 else 'Input composition'),fontsize=9)
        ax.set_xlabel('Mt CO$_2$');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Agriculture: seven contributions hidden inside one sector',x=.03,ha='left',fontsize=14)
    fig.text(.03,.925,'R3 calculation, structural-driver sector; six crop/livestock groups plus the manure-treatment group map to A01',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(FIG/'agriculture_sector_effects.pdf');fig.savefig(FIG/'agriculture_sector_effects.png',dpi=160);plt.close(fig)
    manifest={'analysis_id':'sector-visibility-20260909','validation':{'passed':True,'unique_mapping':True,'agriculture_seven_links_complete':True},
      'inputs':[{'path':str(x.relative_to(ROOT)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in [pp,mp]],
      'mapping_excluded_rows':len(excluded),'units':'kg CO2 inputs; Mt CO2 display','price_basis':'CYP_(t-1) -> PYP_t',
      'limitations':['Fine sector nets are summed over years before sign pooling; this is sector cancellation of period contributions, not cell-level gross activity.',
                    'Reviewed mapping exclusions mean not every coarse parent is fully covered. Agriculture A01 is complete.',
                    'Driver-sector effects alone do not identify a supplying-input sector or a physical substitution.']}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(result[result.coarse_sector=='A01'].to_string(index=False))
    print(ag[['period','effect_id','fine_sector','contribution_Mt']].to_string(index=False))
if __name__=='__main__':main()
