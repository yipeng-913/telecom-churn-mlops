import pandas as pd
import pytest
from telecom_churn.demo import generate


def test_synthetic_generator_reproducible_and_preserves_existing_data(tmp_path):
    a = generate(tmp_path / 'a', 500)
    b = generate(tmp_path / 'b', 500)
    pd.testing.assert_frame_equal(pd.read_csv(a, sep=';'), pd.read_csv(b, sep=';'))
    with pytest.raises(FileExistsError):
        generate(tmp_path / 'a', 500)
