"""Period-preserving annual influence audit; no new SDA is calculated."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
AN = ROOT/'implementation/bilateral_sda_attribution/docs/analysis'
OUT = AN/'results_hypotheses_20260909/temporal'
FIG = ROOT/'src/figures/results/hypotheses_20260909'
PERIODS = {'P1':range(2001,2005),'P2':range(2005,2008),'P3':range(2008,2013)}
NAMES = {'D_v':'Emission intensity','D_A_x':'Technical structure','D_C':'Intermediate sourcing',
         'D_T_coef':'Input composition','D_w_coef':'Intermediate-input share',
         'D_YF':'Final-demand sourcing','D_YB':'Final-demand composition','D_y':'Final-demand volume',
         'delta_g':'Total change','leontief_production_structure':'Leontief effect','final_demand':'Final-demand effect'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load():
    p=ROOT/'data/processed/provisional_panel_v2/cross_year_panel/global_hierarchy.parquet'
    x=pd.read_parquet(p)
    x=x[x.year_current.between(2001,2012)&x.effect_id.ne('emission_intensity')].copy()
    assert x.is_publishable.all() and x.validation_status.eq('passed').all()
    assert x.units.eq('kg CO2').all()
    x=x.rename(columns={'effect_id':'effect','contribution':'net'})
    x['role']='world'; x['group']='World'
    gp=AN/'kyoto_group_deep_dive_20260901/annual_group_effects.csv'
    g=pd.read_csv(gp,dtype={'resolution':str}).rename(columns={'effect_id':'effect','attribution_role':'role','analysis_group':'group'})
    g=g[g.year_current.between(2001,2012)].copy()
    assert g.units.eq('kg CO2').all()
    keep=['year_previous','year_current','resolution','effect','role','group','net']
    a=pd.concat([x[keep],g[keep]],ignore_index=True)
    a['period']=a.year_current.map({y:p for p,years in PERIODS.items() for y in years})
    assert not a.duplicated(keep[:-1]).any()
    assert np.isfinite(a.net).all()
    return a,[p,gp]

def main():
    OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)
    a,inputs=load()
    rows=[]; events=[]
    dims=['resolution','effect','role','group','period']
    for key,q in a.groupby(dims):
        q=q.sort_values('year_current'); years=list(q.year_current); p=key[-1]
        assert years==list(PERIODS[p]), (key,years)
        v=q.net.to_numpy()/1e9; total=v.sum(); mass=np.abs(v).sum(); j=np.argmax(abs(v))
        base=dict(zip(dims,key))
        d={**base,'period_net_Mt':total,'annual_absolute_sum_Mt':mass,'temporal_cancellation':1-abs(total)/mass if mass else 0,
           'dominant_year':years[j],'dominant_year_Mt':v[j],'largest_annual_absolute_share':abs(v[j])/mass if mass else 0,
           'positive_years':int((v>1).sum()),'negative_years':int((v<-1).sum()),'near_zero_years':int((abs(v)<=1).sum())}
        for floor in [0.1,1,10]:
            flips=[y for y,e in zip(years,v) if total*(total-e)<0 and min(abs(total),abs(total-e))>=floor]
            d[f'flip_years_floor_{floor:g}_Mt']=';'.join(map(str,flips))
        rows.append(d)
        for y,e in zip(years,v):
            events.append({**base,'omitted_year':y,'annual_Mt':e,'period_net_Mt':total,'without_year_Mt':total-e,
                           'sign_flip_1Mt':bool(total*(total-e)<0 and min(abs(total),abs(total-e))>=1),
                           'annual_absolute_share':abs(e)/mass if mass else 0})
    summary=pd.DataFrame(rows); loo=pd.DataFrame(events)
    # Reconcile annual group effect panels against independently registered period products.
    pp=AN/'kyoto_period_group_products_20260908/kyoto_period_group_effects.csv'; inputs.append(pp)
    registered=pd.read_csv(pp,dtype={'resolution':str})
    rolemap={'driver':'structural_driver','demanded':'consumer','consumer':'consumer','emitter':'emitter','structural_driver':'structural_driver'}
    ours=a[a.role.ne('world')].copy(); ours['role']=ours.role.map(rolemap)
    checks=ours.groupby(dims,as_index=False).net.sum().merge(registered[dims+['net']],on=dims,suffixes=('_annual','_registered'),validate='one_to_one')
    assert len(checks)>0
    checks['absolute_residual_kg']=abs(checks.net_annual-checks.net_registered)
    checks['relative_residual']=checks.absolute_residual_kg/np.maximum(abs(checks.net_registered),1e9)
    assert checks.relative_residual.max()<1e-5
    a.to_csv(OUT/'annual_effects.csv',index=False)
    summary.to_csv(OUT/'period_influence.csv',index=False)
    loo.to_csv(OUT/'leave_one_year_out.csv',index=False)
    checks.to_csv(OUT/'period_reconciliation.csv',index=False)
    # Trace the selected annual movement through separate driver-country and
    # driver-sector marginals. Do not infer their unavailable joint intersection.
    drill=[]
    for name in ['driver_country','driver_sector']:
        path=ROOT/f'data/processed/provisional_panel_v2/cross_year_panel/{name}.parquet';inputs.append(path)
        data=pd.read_parquet(path)
        data=data[(data.resolution=='R3')&(data.year_current==2003)&(data.effect_id=='D_T_coef')].copy()
        assert data.is_publishable.all() and data.validation_status.eq('passed').all()
        expected=a[(a.role=='world')&(a.resolution=='R3')&(a.year_current==2003)&(a.effect=='D_T_coef')].net.iloc[0]
        assert abs(data.contribution.sum()-expected)/max(abs(expected),1)<1e-5
        data['attribution_grain']=name
        drill.append(data[['attribution_grain','label','contribution']])
    pd.concat(drill).to_csv(OUT/'input_composition_2003_drilldown.csv',index=False)
    # Complete annual paths for all effects on the main target grid, not just selected anomalies.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    effects=['D_v','D_A_x','D_C','D_T_coef','D_w_coef','D_YF','D_YB','D_y']
    fig,axes=plt.subplots(4,2,figsize=(10,11),sharex=True)
    colors=['#e8eef3','#f2ede2','#e9f0e8']
    for ax,e in zip(axes.flat,effects):
        q=a[(a.role=='world')&(a.resolution=='2016')&(a.effect==e)].sort_values('year_current')
        for (p,ys),col in zip(PERIODS.items(),colors): ax.axvspan(min(ys)-.5,max(ys)+.5,color=col,zorder=0)
        vals=q.net.to_numpy()/1e9
        ax.bar(q.year_current,vals,color=np.where(vals>=0,'#b45b36','#287e8d'),width=.7,zorder=2)
        ax.axhline(0,color='#283847',lw=.65); ax.set_title(NAMES[e],loc='left',fontweight='bold')
        ax.set_ylabel('Mt CO$_2$'); ax.set_xticks(range(2001,2013)); ax.tick_params(axis='x',rotation=60)
    fig.suptitle('Annual effects within the three Kyoto periods',fontsize=15,x=.07,ha='left')
    fig.text(.07,.945,'EXIOBASE-derived flows and fossil CO$_2$; 2016-compatible grid | P1: 2001–04   P2: 2005–07   P3: 2008–12',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(FIG/'annual_effect_paths.pdf'); fig.savefig(FIG/'annual_effect_paths.png',dpi=160); plt.close(fig)
    manifest={'analysis_id':'temporal-influence-20260909','units':'kg CO2 inputs; Mt CO2 displays','price_basis':'CYP_(t-1) -> PYP_t',
              'periods':{p:list(y) for p,y in PERIODS.items()},'inputs':[{ 'path':str(p.relative_to(ROOT)),'sha256':sha(p)} for p in inputs],
              'validation':{'passed':True,'period_reconciliations':len(checks),'max_relative_residual':float(checks.relative_residual.max()),
                            'full_annual_coverage':True,'finite_values':True},
              'selection':'complete screening; material sign floor 1 Mt, sensitivity 0.1 and 10 Mt',
              'limitations':['Leave-one-year-out is an influence diagnostic, not an alternative period or causal counterfactual.',
                             'Temporal cancellation uses annual nets, not cell-level gross contributions.'],
              'outputs':[{ 'path':p.name,'sha256':sha(p)} for p in sorted(OUT.glob('*.csv'))]}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Reconciliation:',manifest['validation'])
    print(summary[(summary.role=='world')&(summary.resolution=='2016')][['effect','period','period_net_Mt','dominant_year','largest_annual_absolute_share','flip_years_floor_1_Mt']].to_string(index=False))
if __name__=='__main__': main()
