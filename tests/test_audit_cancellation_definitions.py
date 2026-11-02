import pandas as pd
import pytest
import audit_cancellation_definitions as audit


def test_annual_mean_weighted_and_pooled_cancellation_are_distinct():
    frame=pd.DataFrame([
        {"effect":"D_C","resolution":"2016","year_previous":2000,"year_current":2001,"dimension_level":"overall","net_mt":8.,"positive_mt":9.,"negative_mt":-1.,"gross_mt":10.,"cancellation_ratio":.2},
        {"effect":"D_C","resolution":"2016","year_previous":2001,"year_current":2002,"dimension_level":"overall","net_mt":-2.,"positive_mt":4.,"negative_mt":-6.,"gross_mt":10.,"cancellation_ratio":.8},
    ])
    row=audit.compare(frame).iloc[0]
    assert row.arithmetic_mean_annual_cancellation==.5
    assert row.gross_weighted_annual_cancellation==.5
    assert row.pooled_window_cancellation==.7
    assert row.between_year_sign_cancellation_increment==pytest.approx(.2)


def test_positive_negative_pools_preserve_hidden_cancellation():
    frame=pd.DataFrame([{"effect":"D_YF","resolution":"R3","year_previous":2000,"year_current":2001,"dimension_level":"overall","net_mt":2.,"positive_mt":6.,"negative_mt":-4.,"gross_mt":10.,"cancellation_ratio":.8}])
    row=audit.compare(frame).iloc[0]
    assert row.cumulative_gross_mt==10
    assert row.cumulative_gross_mt!=abs(row.cumulative_net_mt)
