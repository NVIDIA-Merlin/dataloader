import pytest
from testbook import testbook


pytest.importorskip("torch")


@testbook("examples/01b-Getting-started-Pytorch", execute=False)
def test_getting_started_pytorch(tb):
    tb.inject(
        """
        import pandas as pd
        import numpy as np

        !mkdir -p /tmp/ml-25m
        pd.DataFrame({
            'userId': np.random.randint(0, 10, 100_000),
            'movieId': np.random.randint(0, 10, 100_000),
            'rating': np.random.randint(0, 5, 100_000).astype(np.float32)
        }).to_csv('/tmp/ml-25m/ratings.csv', index=False)
        """
    )
    tb.cells[4].source = "DATA_PATH = '/tmp'"
    tb.cells[5].source = ""
    tb.execute()
