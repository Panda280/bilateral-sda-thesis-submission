import pandas as pd
import audit_kyoto_annex_comparison_readiness as audit


def test_inventory_never_promotes_analytical_group_to_legal_status(tmp_path):
    g=tmp_path/"g.csv";c=tmp_path/"c.csv"
    pd.DataFrame([{"canonical_country":"AAA","analysis_group":"Red","presentation_code":"AAA"}]).to_csv(g,index=False)
    pd.DataFrame([{"resolution":"2016","observed_country":"AAA","canonical_country":"AAA"}]).to_csv(c,index=False)
    row=audit.inventory(g,c).iloc[0]
    assert row.unfccc_annex_i_status=="[SOURCE NEEDED]"
    assert row.kyoto_annex_b_cp1_commitment=="[SOURCE NEEDED]"
    assert row.legal_crosswalk_status=="not_ready_do_not_infer"


def test_schema_distinguishes_annex_i_annex_a_and_annex_b():
    schema=pd.DataFrame(audit.SCHEMA,columns=["domain","field","definition","required_authority","local_status"])
    assert "unfccc_annex_i_status" in set(schema.field)
    assert "kyoto_annex_a_scope" in set(schema.field)
    assert "kyoto_annex_b_entry" in set(schema.field)
    annex_a=schema.set_index("field").loc["kyoto_annex_a_scope","definition"]
    assert "not a country-membership flag" in annex_a


def test_required_territorial_and_consumption_fields_are_missing():
    schema=pd.DataFrame(audit.SCHEMA,columns=["domain","field","definition","required_authority","local_status"])
    assert (schema[schema.domain=="territorial"].local_status=="missing_official").all()
    assert (schema[schema.domain=="consumption"].local_status=="missing_local").all()
