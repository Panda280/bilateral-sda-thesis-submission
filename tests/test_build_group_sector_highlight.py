import pandas as pd
import pytest
from scripts.build_group_sector_highlight import build

def test_only_supported_effects_and_sign_stratified_labels():
 rows=[{"effect":"D_C","resolution":"R3","destination_group":"Red","sector_code":"1","all_contributions_kg_co2":10},{"effect":"D_C","resolution":"R3","destination_group":"Red","sector_code":"2","all_contributions_kg_co2":-8}]
 labels=pd.DataFrame([{"resolution":"R3","observed_sector":"1","sector_label":"Agriculture","review_status":"reviewed"},{"resolution":"R3","observed_sector":"2","sector_label":"Mining","review_status":"reviewed"}])
 allrows,selected=build(pd.DataFrame(rows),labels,top_n=1)
 assert len(selected)==2 and set(selected.selection_sign)=={"positive","negative"}
 assert set(allrows.label_status)=={"reviewed_label"} and allrows.result_id.is_unique
 bad=pd.DataFrame([{**rows[0],"effect":"D_A_x"}])
 with pytest.raises(ValueError,match="raw parent"):
  build(bad,labels)
