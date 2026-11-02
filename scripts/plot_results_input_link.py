"""Plot the registered Germany 2011 cereal/cattle input-link case."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'implementation/bilateral_sda_attribution/docs/analysis/results_hypotheses_20260909/trade'
FIG=ROOT/'src/figures/results/hypotheses_20260909'
def main():
    a=pd.read_csv(BASE/'crop_livestock_by_input_changes.csv')
    q=a[(a.year==2011)&(a.destination_country=='DE')&(a.using_sector==4)&(a.input_sector==1)]
    assert len(q)==1;r=q.iloc[0]
    origin=pd.read_csv(BASE/'germany_2011_cereal_cattle_origins.csv'); foreign=origin[origin.relationship=='cross_border']
    assert np.isclose(foreign.change_million_EUR.sum(),r.import_change)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),gridspec_kw={'width_ratios':[1,1.15]})
    pos=np.arange(2);w=.32
    bars=axes[0].bar(pos-w/2,[r.domestic_base,r.cross_border_base],width=w,color='#a8b9c8',label='2010 CYP')
    axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    bars=axes[0].bar(pos+w/2,[r.domestic_t,r.cross_border_t],width=w,color='#287e8d',label='2011 PYP')
    axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=8)
    axes[0].set_xticks(pos,['Domestic','Imported']);axes[0].set_ylabel('Million EUR, 2010 prices');axes[0].set_ylim(0,550);axes[0].legend(frameon=False)
    axes[0].set_title('German cattle/raw-milk cereal inputs',loc='left',fontweight='bold')
    names={'CZ':'Czech Republic','NL':'Netherlands','PL':'Poland','FR':'France'}
    vals=[foreign[foreign.source_country==c].change_million_EUR.iloc[0] for c in names]
    vals.append(foreign.change_million_EUR.sum()-sum(vals));labels=list(names.values())+['Other suppliers, net']
    bars=axes[1].barh(labels,vals,color=['#287e8d']*3+['#b45b36','#a8b9c8'])
    axes[1].invert_yaxis();axes[1].bar_label(bars,fmt='+%.1f',padding=3);axes[1].set_xlim(0,max(vals)*1.2)
    axes[1].set_xlabel('Increase in imported inputs, million EUR');axes[1].set_title('Where the import increase originates',loc='left',fontweight='bold')
    fig.suptitle('A sourcing shift inside the agricultural sector',fontsize=14,x=.02,ha='left')
    fig.text(.02,.915,'R3 joint input link: cereal group → cattle/raw milk | Same-group international suppliers remain cross-border',fontsize=9)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(FIG/'germany_crop_livestock_link.pdf');fig.savefig(FIG/'germany_crop_livestock_link.png',dpi=170);plt.close(fig)
if __name__=='__main__':main()
