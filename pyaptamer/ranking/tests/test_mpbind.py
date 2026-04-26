import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyaptamer.data.loader import MoleculeLoader
from pyaptamer.ranking import MPBind

def test_mpbind_fit_predict():
    # Construct a dummy SELEX dataset
    data = {
        "sequence": ["ATCGAT", "GCTAGC", "ATCGGC", "TTAGCA"],
        "round_0": [100, 50, 10, 5],
        "round_1": [200, 10, 50, 2],
        "round_2": [400, 2, 200, 1],
    }
    df = pd.DataFrame(data)

    model = MPBind(k_mer_len=4, pseudo_count=1.0)
    
    # Fit should return self
    assert model.fit(df) is model

    # meta_z_scores_ should be populated
    assert len(model.meta_z_scores_) > 0

    # Predict on DataFrame
    scores = model.predict(df)
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 4

    # Predict on list
    scores_list = model.predict(data["sequence"])
    assert np.allclose(scores, scores_list)

def test_molecule_loader_csv():
    data = {
        "sequence": ["ATCGAT", "GCTAGC"],
        "round_0": [100, 50],
        "round_1": [200, 10],
    }
    df = pd.DataFrame(data)
    
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df.to_csv(f.name, index=False)
        path = f.name
        
    try:
        loader = MoleculeLoader(path)
        loaded_df = loader.to_df_seq()
        assert "sequence" in loaded_df.columns
        assert "round_0" in loaded_df.columns
        assert "round_1" in loaded_df.columns
        assert loaded_df["round_0"].iloc[0] == 100
    finally:
        os.remove(path)
