"""Tests of the real-slice runner (mptb.realdata.slices) and the Table 14 generator with stub inputs."""
from __future__ import annotations


from mptb.eval.anchors import SpeakerAnchors, TrackAnchors
from mptb.realdata.slices import BACKENDS, run_backend, select_point



def _pt(theta, recall, fpr):
    return {"theta": theta, "recall": recall, "fpr": fpr}


def test_select_point_rules():
    sweep = [_pt(0.3, 0.8, 0.10), _pt(0.5, 0.8, 0.05), _pt(0.7, 0.9, 0.2)]
    assert select_point(sweep, 0.10, tie_fpr=False)["theta"] == 0.3      # first max recall in grid order
    assert select_point(sweep, 0.10, tie_fpr=True)["theta"] == 0.5       # tie broken by lower FPR
    over = select_point([_pt(1, 0.9, 0.4), _pt(2, 0.5, 0.2), _pt(3, 0.6, 0.2)], 0.10, tie_fpr=False)
    assert (over["theta"], over["qualified"]) == (3, False)                # lowest FPR, then highest recall
    assert select_point([_pt(1, None, None)], 0.10, tie_fpr=False) == {"theta": None}


class _Stub:
    """Predictor firing EOT at fixed times; more commits at lower theta."""
    thetas = BACKENDS["smart_turn"].thetas

    def predict(self, audio, sr):
        return {float(th): {"speaker_1": {"eot": [5.0] + ([10.5] if th < 0.5 else []), "interruption": []}}
                for th in self.thetas}


def test_run_backend_sweep_and_selection():
    sa = SpeakerAnchors(eot=TrackAnchors([5.0, 20.0], [(10.0, 12.0)], []),
                        interruption=TrackAnchors([1.0], [], []))
    loaded = []
    entry = run_backend("smart_turn", _Stub(), {"c1": {"speaker_1": sa}},
                        lambda cid, trim: loaded.append((cid, trim)) or {}, log=lambda m: None)
    assert loaded == [("c1", True)]
    assert entry["eot_only"] is True and list(entry["sweep"]) == ["eot"]
    by_theta = {d["theta"]: d for d in entry["sweep"]["eot"]}
    assert by_theta[0.3]["fpr"] == 1.0 and by_theta[0.5]["fpr"] == 0.0 and by_theta[0.5]["recall"] == 0.5
    assert by_theta[0.5]["n_pos"] == 2 and by_theta[0.5]["n_neg"] == 1
    chosen = entry["tracks"]["eot"]
    assert (chosen["theta"], chosen["qualified"]) == (0.5, True)
