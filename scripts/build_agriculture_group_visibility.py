"""Validate joint driver-country/sector totals and report agriculture by group.

Uses the saved GWP100 campaign (R-2026-053), retaining its adjacent-year
price basis. No new SDA is calculated. Non-Reduction remains in controls.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'implementation/bilateral_sda_attribution/docs/analysis/agriculture_groups_20260914'
CON = ROOT/'references/current_codebase/concordance'
GROUPS = {'Red': 'Reduction', 'Non-Participant': 'Non-Participation'}
EFFECTS = {'D_A_x': 'Technical structure', 'D_C': 'Sourcing', 'D_T_coef': 'Input composition', 'D_w_coef': 'Intermediate-input share'}
SECTORS = ['1', '2', '3', '4', '5', '6', '97']
LABELS = ['Rice/wheat/cereals', 'Vegetables/oil seeds', 'Other crops', 'Cattle/raw milk',
          'Pigs/poultry', 'Other animal products', 'Manure treatment']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
    source = OUT/'extracted_joint_data.json'
    raw = json.loads(source.read_text(encoding='utf-8-sig'))
    p = pd.DataFrame(raw['rows'], columns=raw['columns'])
    ctl = pd.DataFrame(raw['controls'], columns=raw['control_columns'])
    checks = []
    inputs = [source]
    # Independently reproduce BOTH published marginals from full joint arrays.
    for role in ['driver_country', 'driver_sector']:
        path = ROOT/f'data/processed/provisional_panel_v2/cross_year_panel/{role}.parquet'
        inputs.append(path)
        panel = pd.read_parquet(path)
        panel = panel[panel.resolution.isin(['R3','2016']) & panel.year_current.between(2001,2012) & panel.effect_id.isin(['D_C','D_A_x','D_T_coef','D_w_coef'])]
        assert panel.validation_status.eq('passed').all() and panel.is_publishable.all()
        joint = ctl[ctl.role.eq(role)]
        keys = ['resolution','year_current','effect_id','label']
        m = joint.merge(panel, on=keys, validate='one_to_one', suffixes=('_new','_old'), how='outer', indicator=True)
        assert m['_merge'].eq('both').all()
        diff = m.contribution_new - m.contribution_old
        assert np.allclose(m.contribution_new, m.contribution_old, rtol=1e-9, atol=.1), (role, abs(diff).max())
        checks.append({'check': role+' panel reconciliation', 'max_abs_error_kg': float(abs(diff).max()), 'passed': True})
    cpath = CON/'normalized_2026-08-31/country_concordance_by_resolution.csv'
    gpath = CON/'kyoto_country_groups.csv'
    mpath = CON/'normalized_2026-08-31/granularity_sector_mapping_review.csv'
    inputs += [cpath,gpath,mpath]
    countries = pd.read_csv(cpath, dtype=str)
    registry = pd.read_csv(gpath, dtype=str)
    assert not registry.country_code.duplicated().any()
    mapping = countries.merge(registry[['country_code','analysis_group']], left_on='canonical_country',right_on='country_code',how='left',validate='many_to_one')
    assert mapping.analysis_group.notna().all()
    p = p.merge(mapping[['resolution','observed_country','analysis_group']],left_on=['resolution','country'],right_on=['resolution','observed_country'],how='left',validate='many_to_one')
    assert p.analysis_group.notna().all()
    sector_map = pd.read_csv(mpath,dtype=str)
    a01 = sector_map[(sector_map.coarse_resolution=='2016') & (sector_map.coarse_sector=='A01') & (sector_map.mapping_status=='reviewed_unique')]
    assert set(a01.fine_sector)==set(SECTORS) and not a01.fine_sector.duplicated().any()
    assert set(p[p.resolution.eq('R3')].sector)==set(SECTORS)
    p['period'] = p.year_current.map(lambda y: 'P1' if y<=2004 else 'P2' if y<=2007 else 'P3')
    p.to_csv(OUT/'annual_joint_agriculture.csv', index=False)
    fine = p.groupby(['resolution','period','effect_id','analysis_group','sector'],as_index=False).contribution.sum()
    fine['contribution_Mt'] = fine.contribution/1e9
    fine.to_csv(OUT/'period_group_sector_effects.csv',index=False)
    # Reconcile the sum of ALL THREE groups to the original global illustration.
    original_path = ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909/sectors/fine_sector_period_effects.csv'
    inputs.append(original_path)
    original = pd.read_csv(original_path,dtype={'fine_sector':str})
    original = original[original.coarse_sector.eq('A01')]
    summed = fine[fine.resolution.eq('R3')].groupby(['period','effect_id','sector'],as_index=False).contribution.sum()
    m = summed.merge(original, left_on=['period','effect_id','sector'],right_on=['period','effect_id','fine_sector'],validate='one_to_one',suffixes=('_new','_old'),how='outer',indicator=True)
    assert m['_merge'].eq('both').all()
    assert np.allclose(m.contribution_new,m.contribution_old,rtol=1e-9,atol=.1)
    checks.append({'check':'all three groups reconcile to original global agricultural results','passed':True,'max_abs_error_kg':float(abs(m.contribution_new-m.contribution_old).max())})
    summaries=[]
    for (period,effect,group),q in fine[fine.resolution.eq('R3')].groupby(['period','effect_id','analysis_group']):
        assert len(q)==7
        v=q.contribution_Mt
        coarse=fine[(fine.resolution=='2016')&(fine.period==period)&(fine.effect_id==effect)&(fine.analysis_group==group)]
        assert len(coarse)==1
        summaries.append(dict(period=period,effect_id=effect,analysis_group=group,mapped_R3_Mt=v.sum(),positive_Mt=v[v>0].sum(),negative_Mt=v[v<0].sum(),cancellation_pct=100*(1-abs(v.sum())/abs(v).sum()) if abs(v).sum() else 0,recalculated_2016_Mt=coarse.contribution_Mt.iloc[0]))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'group_aggregation_comparison.csv',index=False)
    hierarchy=fine.pivot(index=['resolution','period','analysis_group','sector'],columns='effect_id',values='contribution_Mt')
    agriculture_residual=hierarchy.D_A_x-hierarchy.D_T_coef-hierarchy.D_w_coef
    report={'reaggregation_validation_passed':True,'checks':checks,'source_hierarchy_residuals':raw['hierarchy_checks'],
            'source_hierarchy_strict_identity_check_passed':False,
            'source_hierarchy_note':'Initial strict identity test failed (rtol=1e-9, atol=0.01 kg). Residuals are inherited from saved arrays; no causal or numerical explanation established here and no correction applied.',
            'max_agriculture_group_sector_period_hierarchy_residual_Mt':float(abs(agriculture_residual).max()),
            'inputs':[{'path':str(p.relative_to(ROOT)), 'sha256':sha(p)} for p in inputs],
            'indicator':'GWP100, kg CO2-equivalent; display Mt CO2-equivalent (R-2026-053). Legacy panel kg CO2 labels are incorrect.',
            'role':'structural-driver country and structural-driver sector jointly; all emitting countries/sectors summed',
            'groups_displayed':list(GROUPS.values()),'all_groups_retained_for_reconciliation':True,
            'limitations':['Period sector nets, not gross matrix-cell or annual cancellation.',
                           'Group assignment uses driver country, not emitter country.',
                           'Independent 2016 comparison changes country and sector resolution together.',
                           'Parent and child effects are reported separately; source hierarchy residuals retained without correction.']}
    (OUT/'validation_manifest.json').write_text(json.dumps(report,indent=2))
    print(summary[summary.analysis_group.isin(GROUPS)&summary.effect_id.isin(EFFECTS)].round(3).to_string(index=False))
    print('Hierarchy residuals:',raw['hierarchy_checks'])
    if '--figures' in sys.argv:
        figures(fine,summary)

def figures(fine,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=ROOT/'src/figures/results/hypotheses_20260909'
    # Equal effect-specific scales across groups and periods support comparison.
    limits={}
    plot_effects = ['D_C', 'D_T_coef', 'D_w_coef']
    plot_titles = ['Intermediate sourcing', 'Input composition', 'Intermediate-input share']
    for effect in plot_effects:
        v=fine[(fine.resolution=='R3')&fine.analysis_group.isin(GROUPS)&(fine.effect_id==effect)].contribution_Mt
        lo,hi=min(0,v.min()),max(0,v.max()); pad=(hi-lo)*.08
        limits[effect]=(lo-pad,hi+pad)
    for group,name in GROUPS.items():
        fig,axes=plt.subplots(3,3,figsize=(11,8.6),sharex='row')
        for r,effect in enumerate(plot_effects):
            for c,period in enumerate(['P1','P2','P3']):
                q=fine[(fine.resolution=='R3')&(fine.analysis_group==group)&(fine.effect_id==effect)&(fine.period==period)].set_index('sector').loc[SECTORS]
                v=q.contribution_Mt.to_numpy(); ax=axes[r,c]
                ax.barh(range(7),v,color=np.where(v>=0,'#b45b36','#287e8d'))
                ax.axvline(0,color='#283847',lw=.7)
                ax.set_yticks(range(7),LABELS if c==0 else ['']*7);ax.invert_yaxis()
                ax.set_xlim(*limits[effect]);ax.set_title(period+' | '+plot_titles[r],fontsize=10)
                ax.set_xlabel('Mt CO$_2$-equivalent');ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Agriculture: '+name, x=.03,ha='left',fontsize=15)
        fig.text(.03,.94,'R3 structural-driver country and sector; all emitting countries included',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.915))
        stem='agriculture_sector_effects_'+('reduction' if group=='Red' else 'non_participation')
        fig.savefig(figdir/(stem+'.pdf'));fig.savefig(figdir/(stem+'.png'),dpi=160);plt.close(fig)
    lines=[r'\begin{table}[htbp]\centering\small',
           r'\caption{Agriculture (A01) by structural-driver country group: summed R3 contributions and independently calculated 2016-grid contributions, in Mt CO$_2$-equivalent. Cancellation is measured across the seven R3 sector period nets. Input composition and intermediate-input share are children of technical structure; the parent must not be added to its children. Sourcing is reported separately.}',
           r'\label{tab:agriculture-aggregation}',r'\begin{tabular}{llrrr}\toprule',
           r'Period & Effect & R3 sum & Cancellation (\%) & 2016 grid\\\midrule']
    for group,name in GROUPS.items():
        lines.append(r'\multicolumn{5}{l}{\textit{'+name+r'}}\\')
        for period in ['P1','P2','P3']:
            for effect,label in EFFECTS.items():
                row=summary[(summary.analysis_group==group)&(summary.period==period)&(summary.effect_id==effect)].iloc[0]
                lines.append(f'{period} & {label} & {row.mapped_R3_Mt:+.1f} & {row.cancellation_pct:.0f} & {row.recalculated_2016_Mt:+.1f} '+r'\\')
    lines.append(r'\bottomrule\end{tabular}\end{table}')
    (ROOT/'src/chapters/results_revised/generated/tab_agriculture_aggregation_groups.tex').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    build()
