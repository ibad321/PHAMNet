"""CPU-friendly tests that do not need the dataset or pretrained weights."""
import numpy as np
import pytest

from phamnet.config import PRESETS, VARIANTS, TrainConfig
from phamnet.data import leakage, make_split
from phamnet.duplicates import group_ids
from phamnet.metrics import clopper_pearson, evaluate, mcnemar_exact


def _labels(n=4217):
    rng = np.random.default_rng(0)
    return rng.choice(4, size=n, p=[0.24, 0.25, 0.24, 0.27])


def test_split_sizes_match_study():
    labels = _labels()
    split = make_split(labels, np.arange(len(labels)))
    assert (len(split.train), len(split.val), len(split.test)) == (3375, 420, 422)
    assert not (set(split.train) & set(split.test)) and not (set(split.train) & set(split.val))


def test_duplicate_aware_split_has_no_leakage():
    labels = _labels(1000)
    groups = np.arange(1000) // 2          # consecutive pairs are near-duplicates
    split = make_split(labels, groups, duplicate_aware=True)
    assert leakage(groups, split.train, split.test) == 0
    assert leakage(groups, split.train, split.val) == 0


def test_group_ids_threshold():
    pairs = [(0, 1, 0.99), (1, 2, 0.90), (3, 4, 0.96)]
    g = group_ids(5, pairs, 0.95)
    assert g[0] == g[1] and g[3] == g[4] and g[2] != g[1]


def test_clopper_pearson_contains_estimate():
    lo, hi = clopper_pearson(404, 422)
    assert lo < 404 / 422 < hi


def test_mcnemar_identical_predictions():
    c = np.array([True, False, True, True])
    assert mcnemar_exact(c, c) == (0, 0, 1.0)


def test_evaluate_perfect_predictions():
    y = np.repeat(np.arange(4), 10)
    probs = np.eye(4)[y]
    out = evaluate(y, probs, ["a", "b", "c", "d"], resamples=50)
    assert out["accuracy"] == 1.0 and out["correct"] == 40 and out["macro_auroc"] == 1.0


def test_config_roundtrip(tmp_path):
    cfg = TrainConfig(epochs=3, runs=["phamnet"], seeds=[1, 2])
    path = tmp_path / "c.json"
    cfg.to_json(str(path))
    assert TrainConfig.from_json(str(path)) == cfg


def test_presets_refer_to_known_variants():
    for runs, _ in PRESETS.values():
        assert all(r in VARIANTS for r in runs)


tf = pytest.importorskip("tensorflow")


@pytest.mark.parametrize("mode", ["sab", "gcab", "ham", "ham_fixed", "cbam", "bam"])
def test_attention_preserves_shape(mode):
    from phamnet.layers import Attention

    x = tf.random.normal([2, 7, 7, 64])
    assert Attention(64, mode)(x).shape == x.shape


def test_ham_alpha_beta_trainable_only_for_ham():
    from phamnet.layers import Attention

    x = tf.random.normal([1, 7, 7, 32])
    ham, fixed = Attention(32, "ham"), Attention(32, "ham_fixed")
    ham(x), fixed(x)
    assert len(ham.trainable_weights) > len(fixed.trainable_weights)
    assert float(ham.alpha) == 0.5 and float(ham.beta) == 0.5


def test_mffs_output_shape():
    from tensorflow.keras import layers as L

    from phamnet.layers import mffs

    inp = L.Input((32, 32, 3))
    assert mffs(inp).shape[1:] == (32, 32, 3)


def test_activity_l1_is_batch_mean():
    from phamnet.layers import ActivityL1

    layer = ActivityL1(0.5)
    layer(tf.ones([8, 4]))
    assert float(layer.losses[0]) == pytest.approx(0.5 * 4)
