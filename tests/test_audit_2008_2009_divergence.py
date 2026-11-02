import pandas as pd
import pytest
import audit_2008_2009_divergence as audit


def synthetic():
    rows=[]
    vals={"D_v":10,"D_C":3,"D_T_coef":-8,"D_w_coef":1,"D_YF":2,"D_YB":-4,"D_y":1}
    vals|={"leontief_production_structure":-4,"final_demand":-1,"delta_g":5}
    for resolution,scale in [("2013",1),("2016",2),("R3",3)]:
        for effect,value in vals.items():
            net=value*scale;gross=abs(net)+2
            rows.append({"year_previous":2008,"year_current":2009,"resolution":resolution,"units":"kg CO2","effect_id":effect,"contribution":net*1e9,"positive_contribution":((gross+net)/2)*1e9,"negative_contribution":((net-gross)/2)*1e9,"gross_contribution":gross*1e9})
    return pd.DataFrame(rows)


def test_reconstructs_primary_hierarchy_and_keeps_sourcing_separate():
    ep=audit.episode(synthetic());r=audit.reconciliation(ep).set_index("resolution").loc["2013"]
    assert r.leontief_parent_reconstructed==-4
    assert r.final_demand_parent_reconstructed==-1
    assert r.total_reconstructed==5 and r.total_residual==0
    assert r.D_C_sourcing==3 and r.D_YF_sourcing==2


def test_pairwise_grid_difference_is_descriptive_composite():
    pairs=audit.pairwise(audit.episode(synthetic()))
    row=pairs[(pairs.effect_id=="D_C")&(pairs.resolution_left=="2013")&(pairs.resolution_right=="R3")].iloc[0]
    assert row.signed_difference_right_minus_left==6
    assert "composite" in row.comparison_class


def test_availability_does_not_claim_database_or_causal_identification():
    a=audit.availability().set_index("question")
    assert not a.loc["Pure WIOD-versus-EXIOBASE database effect","available"]
    assert not a.loc["Economic or crisis cause of the episode","available"]
