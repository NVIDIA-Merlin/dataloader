import os
import pytest
from testbook import testbook

pytest.importorskip("tensorflow")

pytestmark = pytest.mark.tensorflow


@testbook("examples/02-Multi-GPU-Tensorflow-with-Horovod.ipynb", execute=False)
def test_getting_started_tensorflow(tb):
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
    tb.cells[7].source.replace("GPU_COUNT = 2", "GPU_COUNT = 1")
    tb.execute()
    os.system("horovodrun -np 1 python tf_trainer.py")
