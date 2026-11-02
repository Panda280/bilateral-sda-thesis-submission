"""Descriptive country-year associations and matched-price crop/livestock links."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
RAW=BASE/'trade_inputs'; OUT=BASE/'trade'; FIG=ROOT/'src/figures/results/hypotheses_20260909'
REGIONS={'WA','WE','WF','WL','WM','ROW','RoW','WT'}
def period(y):return 'P1' if y<=2004 else 'P2' if y<=2007 else 'P3'
def twoway(v,country,year):
    s=pd.Series(np.asarray(v));df=pd.DataFrame({'v':s,'c':np.asarray(country),'t':np.asarray(year)})
    return (df.v-df.groupby('c').v.transform('mean')-df.groupby('t').v.transform('mean')+df.v.mean()).to_numpy()
def estimate(q,xcol,ycol='delta_territorial_Mt'):
    x=q[xcol].to_numpy();y=q[ycol].to_numpy();n=len(q);g=q.country.nunique();nt=q.year.nunique()
    assert n==g*nt and g>2
    xw=twoway(x,q.country,q.year);yw=twoway(y,q.country,q.year)
    beta=np.dot(xw,yw)/np.dot(xw,xw);u=yw-beta*xw
    scores=pd.DataFrame({'country':q.country.to_numpy(),'s':xw*u}).groupby('country').s.sum()
    se=np.sqrt(np.dot(scores,scores)/(np.dot(xw,xw)**2)*g/(g-1)*(n-1)/(n-g-nt))
    crit=stats.t.ppf(.975,g-1)
    return {'n':n,'countries':g,'pearson':stats.pearsonr(x,y).statistic,'spearman':stats.spearmanr(x,y).statistic,
            'two_way_slope':beta,'cluster_se':se,'ci_low':beta-crit*se,'ci_high':beta+crit*se,
            'two_way_partial_r':stats.pearsonr(xw,yw).statistic}
def main():
    OUT.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((RAW/'analysis_manifest.json').read_text()); assert manifest['completed']
    raw=pd.read_csv(RAW/'country_levels_and_trade.csv')
    assert not raw.duplicated(['grid','year','price_basis','country']).any()
    cyp=raw[raw.price_basis=='CYP'].copy();pyp=raw[raw.price_basis=='PYP'].copy()
    assert cyp.groupby(['grid','country']).size().eq(13).all()
    cyp=cyp.sort_values(['grid','country','year'])
    for old,new in [('territorial_kg_CO2','delta_territorial_Mt'),('imported_footprint_kg_CO2','delta_imported_footprint_Mt'),('effective_territorial_kg_CO2','delta_effective_territorial_Mt')]:
        cyp[new]=cyp.groupby(['grid','country'])[old].diff()/1e9
    prev=raw[raw.price_basis=='CYP'][['grid','year','country','intermediate_imports_million_EUR','final_imports_million_EUR']].copy()
    prev.year+=1
    trade=pyp.merge(prev,on=['grid','year','country'],suffixes=('_t','_previous'),validate='one_to_one')
    trade['delta_import_expenditure_billion_EUR']=(trade.intermediate_imports_million_EUR_t+trade.final_imports_million_EUR_t-trade.intermediate_imports_million_EUR_previous-trade.final_imports_million_EUR_previous)/1000
    a=cyp[cyp.year>=2001].merge(trade[['grid','year','country','delta_import_expenditure_billion_EUR']],on=['grid','year','country'],validate='one_to_one')
    assert np.isfinite(a[['delta_territorial_Mt','delta_imported_footprint_Mt','delta_import_expenditure_billion_EUR']]).all().all()
    a['period']=a.year.map(period);a['aggregate_region']=a.country.isin(REGIONS)
    a.to_csv(OUT/'annual_country_changes.csv',index=False)
    results=[]; influence=[]
    for grid,q0 in a.groupby('grid'):
      q0=q0[~q0.aggregate_region]
      for scope,q in [('All countries',q0)]+[(g,d) for g,d in q0.groupby('group') if d.country.nunique()>3]:
        for x in ['delta_imported_footprint_Mt','delta_import_expenditure_billion_EUR']:
          e=estimate(q,x); results.append({'grid':grid,'scope':scope,'x':x,**e})
          for country in sorted(q.country.unique()):
            ee=estimate(q[q.country!=country],x)
            influence.append({'grid':grid,'scope':scope,'x':x,'omitted_country':country,**ee})
    pd.DataFrame(results).to_csv(OUT/'associations.csv',index=False)
    pd.DataFrame(influence).to_csv(OUT/'leave_one_country_out.csv',index=False)
    sums=cyp.groupby(['grid','group','year'],as_index=False)[['territorial_kg_CO2','imported_footprint_kg_CO2','domestic_footprint_kg_CO2','total_footprint_kg_CO2']].sum()
    changes=[]
    for (grid,g),q in sums.groupby(['grid','group']):
      q=q.set_index('year')
      for p,start,end in [('P1',2000,2004),('P2',2004,2007),('P3',2007,2012)]:
        changes.append({'grid':grid,'group':g,'period':p,'start':start,'end':end,
            **{c.replace('_kg_CO2','_change_Mt'):(q.loc[end,c]-q.loc[start,c])/1e9 for c in q.columns if c.endswith('_kg_CO2')},
            'territorial_change_percent':100*(q.loc[end,'territorial_kg_CO2']/q.loc[start,'territorial_kg_CO2']-1),
            'imported_footprint_change_percent':100*(q.loc[end,'imported_footprint_kg_CO2']/q.loc[start,'imported_footprint_kg_CO2']-1)})
    pd.DataFrame(changes).to_csv(OUT/'period_group_level_changes.csv',index=False)
    # Reconstructed input links: 3 crop supplier sectors into each livestock sector/country.
    links=pd.read_csv(RAW/'crop_livestock_links.csv',dtype={'input_sector':str,'using_sector':str})
    assert np.isfinite(links.million_EUR).all()
    keys=['year','price_basis','destination_country','destination_group','using_sector']
    l=links.groupby(keys+['relationship'],as_index=False).million_EUR.sum().pivot(index=keys,columns='relationship',values='million_EUR').reset_index()
    assert l[['domestic','cross_border']].notna().all().all()
    l['total']=l.domestic+l.cross_border;l['import_share']=l.cross_border/l.total.replace(0,np.nan)
    prev=l[l.price_basis=='CYP'].copy();prev.year+=1
    d=l[l.price_basis=='PYP'].merge(prev,on=['year','destination_country','destination_group','using_sector'],suffixes=('_t','_base'),validate='one_to_one')
    for k in ['domestic','cross_border','total','import_share']:d['change_'+k]=d[k+'_t']-d[k+'_base']
    d['period']=d.year.map(period)
    d['strict_substitution_signature']=(d.change_domestic < -1)&(d.change_cross_border>1)&(d.change_import_share>0)&(d.total_base>0)&(d.total_t>0)
    d['matched_offset_million_EUR']=np.minimum(-d.change_domestic,d.change_cross_border).clip(lower=0).where(d.strict_substitution_signature,0)
    d.to_csv(OUT/'annual_crop_livestock_changes.csv',index=False)
    rank=d[d.strict_substitution_signature].sort_values('matched_offset_million_EUR',ascending=False)
    rank.to_csv(OUT/'crop_livestock_signature_cases.csv',index=False)
    # Keep the supplying crop sector as well as the livestock user; this tests a joint link.
    keys2=keys+['input_sector']
    li=links.groupby(keys2+['relationship'],as_index=False).million_EUR.sum().pivot(index=keys2,columns='relationship',values='million_EUR').reset_index()
    prior=li[li.price_basis=='CYP'].copy();prior.year+=1
    di=li[li.price_basis=='PYP'].merge(prior,on=['year','destination_country','destination_group','using_sector','input_sector'],suffixes=('_t','_base'),validate='one_to_one')
    di['domestic_change']=di.domestic_t-di.domestic_base;di['import_change']=di.cross_border_t-di.cross_border_base
    di['strict_signature']=(di.domestic_change < -1)&(di.import_change>1)
    di['near_zero_domestic_end']=di.domestic_t.abs()<.01*di.domestic_base.abs()
    di['near_zero_import_base']=di.cross_border_base.abs()<.01*di.cross_border_t.abs()
    di['period']=di.year.map(period)
    di.to_csv(OUT/'crop_livestock_by_input_changes.csv',index=False)
    named=d[~d.destination_country.isin(REGIONS)].groupby('period').strict_substitution_signature.agg(['sum','count']).reset_index()
    named.to_csv(OUT/'national_signature_counts.csv',index=False)
    # Trace the Japan 2008 footprint example by actual emitter origin.
    pair=pd.read_csv(RAW/'embodied_country_pairs.csv')
    qp=pair[(pair.grid=='R3')&(pair.destination=='JP')&pair.year.isin([2007,2008])&(pair.relationship=='cross_border')]
    qp=qp.pivot(index=['origin','origin_group'],columns='year',values='embodied_kg_CO2').reset_index()
    qp['change_Mt']=(qp[2008]-qp[2007])/1e9
    qp.sort_values('change_Mt',ascending=False).to_csv(OUT/'japan_2008_import_origins.csv',index=False)
    # A detailed cereal-to-cattle example with no near-zero endpoint.
    selected=links[(links.destination_country=='DE')&(links.using_sector=='4')&(links.input_sector=='1')&
                   (((links.year==2010)&(links.price_basis=='CYP'))|((links.year==2011)&(links.price_basis=='PYP')))]
    by_origin=selected.pivot(index=['source_country','source_group','relationship'],columns='year',values='million_EUR').reset_index()
    by_origin['change_million_EUR']=by_origin[2011]-by_origin[2010]
    by_origin.to_csv(OUT/'germany_2011_cereal_cattle_origins.csv',index=False)
    # Show levels without confusing domestic footprint with territorial emissions.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,g in zip(axes,['Red','Non-Red','Non-Participant']):
      q=sums[(sums.grid=='R3')&(sums.group==g)]
      for k,label,col in [('territorial_kg_CO2','Territorial','#283847'),('imported_footprint_kg_CO2','Imported footprint','#b45b36'),('domestic_footprint_kg_CO2','Domestic footprint','#287e8d')]:
        ax.plot(q.year,q[k]/1e12,label=label,color=col,lw=1.8)
      ax.set_title({'Red':'Reduction','Non-Red':'Non-Reduction','Non-Participant':'Non-Participation'}[g]);ax.set_ylabel('Gt CO$_2$');ax.set_xticks([2000,2004,2007,2012]);ax.grid(axis='y',alpha=.2)
    axes[0].legend(frameon=False,fontsize=8);fig.suptitle('Territorial emissions and the origin of consumption footprints',fontsize=13)
    fig.tight_layout();fig.savefig(FIG/'territorial_imported_levels.pdf');fig.savefig(FIG/'territorial_imported_levels.png',dpi=160);plt.close(fig)
    validation=pd.read_csv(RAW/'footprint_validation.csv')
    m={'analysis_id':'trade-associations-20260909','source_manifest_sha256':hashlib.sha256((RAW/'analysis_manifest.json').read_bytes()).hexdigest(),
       'validation':{'passed':True,'country_year_coverage_complete':True,'max_relative_effective_mass_residual':validation.relative_country_emission_balance.max(),
                     'max_raw_impact_loss_relative':validation.impact_loss_relative.max()},
       'limitations':['Associations are descriptive, with country and year effects and country-clustered intervals; not causal.',
                     'Territorial totals use raw emissions. Footprints omit emissions at nonpositive output; the loss is quantified.',
                     'Import expenditure compares PYP_t with CYP_(t-1), not physical import volume.',
                     'Strict crop/livestock signature uses a 1 million EUR minimum change on both sides; not a measured physical replacement.'],
       'outputs':[{'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(OUT.glob('*.csv'))]}
    (OUT/'analysis_manifest.json').write_text(json.dumps(m,indent=2))
    print(pd.DataFrame(results).to_string(index=False));print(pd.DataFrame(changes).query("grid=='R3'").to_string(index=False)); print(rank.head(12).to_string(index=False));print(m['validation'])
if __name__=='__main__':main()
