import os
import subprocess

import numpy as np
import pandas as pd
import pytest
from testbook import testbook

pytest.importorskip("tensorflow")

pytestmark = pytest.mark.tensorflow


@pytest.mark.multigpu
@testbook("examples/02-Multi-GPU-Tensorflow-with-Horovod.ipynb", execute=False, timeout=120)
def test_getting_started_tensorflow(tb, tmpdir):
    ml_25m_dir = tmpdir / "ml-25"
    ml_25m_dir.mkdir()
    ratings_path = ml_25m_dir / "ratings.csv"
    pd.DataFrame(
        {
            "userId": np.random.randint(0, 10, 100_000),
            "movieId": np.random.randint(0, 10, 100_000),
            "rating": np.random.randint(0, 5, 100_000).astype(np.float32),
        }
    ).to_csv(ratings_path, index=False)

    tb.inject(
        f"""
        import os
        os.environ["DATA_PATH"] = "{str(tmpdir)}"
        """
    )

    tb.execute()

    curr_path = os.path.abspath(__file__)
    repo_root = os.path.relpath(os.path.normpath(os.path.join(curr_path, "../../..")))
    hvd_wrap_path = os.path.join(repo_root, "merlin/dataloader/utils/tf/hvd_wrapper.sh")
    with subprocess.Popen(
        [
            "horovodrun",
            "-np",
            "2",
            "-H",
            "localhost:2",
            "sh",
            hvd_wrap_path,
            "python",
            "tf_trainer.py",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        process.wait()
        stdout, stderr = process.communicate()
        print(stdout, stderr)
        assert "Loss" in str(stdout)

    assert any(f.startswith("checkpoints-") for f in os.listdir(os.getcwd()))
