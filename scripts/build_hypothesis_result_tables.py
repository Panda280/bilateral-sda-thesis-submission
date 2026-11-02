"""Generate focused chapter tables from the registered hypothesis analyses."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909'
OUT=ROOT/'src/chapters/results_revised/generated'
def main():
    x=pd.read_csv(BASE/'temporal/leave_one_year_out.csv',dtype={'resolution':str})
    rows=[]
    for effect,p,y,label in [('D_T_coef','P1',2003,'Input composition'),('D_A_x','P2',2006,'Technical structure'),('D_YB','P3',2010,'Demand composition')]:
      for grid in ['2013','2016','R3']:
        q=x[(x.resolution==grid)&(x.role=='world')&(x.effect==effect)&(x.period==p)&(x.omitted_year==y)]
        assert len(q)==1;r=q.iloc[0]
        rows.append(f'{label}, {p} & {grid} & {y} & {r.period_net_Mt:+.0f} & {r.annual_Mt:+.0f} & {r.without_year_Mt:+.0f} '+r'\\')
    (OUT/'tab_annual_influence.tex').write_text('\n'.join([
      r'\begin{table}[htbp]\centering\small',r'\setlength{\tabcolsep}{4pt}',
      r'\caption{Annual sign influence at world level (Mt CO$_2$). The last column subtracts the indicated year from the complete period; it is an influence diagnostic, not a replacement result period. Same EXIOBASE sources, three target grids (R-2026-042).}\label{tab:annual-influence}',
      r'\begin{tabular}{llrrrr}\toprule',r'Effect, period & Grid & Year & Period & Year effect & Without year\\\midrule',*rows,r'\bottomrule\end{tabular}\end{table}'])+'\n')
    a=pd.read_csv(BASE/'sectors/within_parent_visibility.csv');a=a[a.coarse_sector=='A01']
    rows=[]
    for p in ['P1','P2','P3']:
      for e,label in [('D_A_x','Technical structure'),('D_C','Sourcing'),('D_T_coef','Input composition')]:
        q=a[(a.period==p)&(a.effect==e)];assert len(q)==1;r=q.iloc[0]
        rows.append(f'{p} & {label} & {r.mapped_R3_net_Mt:+.1f} & {100*r.sector_cancellation:.0f} & {r.recalculated_2016_net_Mt:+.1f} '+r'\\')
    (OUT/'tab_agriculture_aggregation.tex').write_text('\n'.join([
      r'\begin{table}[htbp]\centering\small',
      r'\caption{Agriculture (A01): reporting aggregation and independent target-grid recalculation. Contributions in Mt CO$_2$; cancellation is a percentage of the sum of absolute fine-sector period nets. The parent technical-structure effect is not added to its input-composition child (R-2026-043).}\label{tab:agriculture-aggregation}',
      r'\begin{tabular}{llrrr}\toprule',r'Period & Effect & Mapped R3 & Cancellation (\%) & Recalculated 2016\\\midrule',*rows,r'\bottomrule\end{tabular}\end{table}'])+'\n')
    if (BASE/'trade/associations.csv').exists():
      a=pd.read_csv(BASE/'trade/associations.csv');rows=[]
      for grid in ['WIOT2013','WIOT2016','R3']:
       for scope in ['All countries','Red']:
        q=a[(a.grid==grid)&(a.scope==scope)&(a.x=='delta_imported_footprint_Mt')];assert len(q)==1;r=q.iloc[0]
        rows.append(f'{grid.replace("WIOT","")} & {"Reduction" if scope=="Red" else scope} & {r.countries:.0f} & {r.pearson:+.2f} & {r.spearman:+.2f} & {r.two_way_slope:+.2f} & [{r.ci_low:+.2f}, {r.ci_high:+.2f}] '+r'\\')
      (OUT/'tab_trade_associations.tex').write_text('\n'.join([
       r'\begin{table}[htbp]\centering\small\setlength{\tabcolsep}{4pt}',
       r'\caption{Annual territorial-emission changes and imported-footprint changes, 2001--2012. Both variables are in Mt CO$_2$. The slope includes country and year effects; intervals are 95\% country-clustered intervals. Aggregate rest-of-world regions are excluded from estimation. All grids use EXIOBASE-derived inputs (R-2026-045).}\label{tab:trade-associations}',
       r'\begin{tabular}{llrrrrl}\toprule',r'Grid & Sample & Countries & Pearson & Spearman & Slope & Interval\\\midrule',*rows,r'\bottomrule\end{tabular}\end{table}'])+'\n')
if __name__=='__main__':main()
