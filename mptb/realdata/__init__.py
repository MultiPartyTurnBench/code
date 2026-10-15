"""Real-audio validation slices of the paper (Table 14, Appendix G.1 and H.2).

The offline baselines and the MultiPartyTurnBench scorer are applied to two real-speech corpora:

* :mod:`mptb.realdata.turnbench` - a 12-conversation dyadic (N=2) slice of the TurnBench dev set, with
  gold rebuilt from TurnBench's three annotator tracks (2-of-3 consensus, +/-200 ms endpoints);
* :mod:`mptb.realdata.ami` - 15 four-party (N=4) AMI meetings, with floor events derived from AMI's
  manual segment and word annotations by a documented floor-state machine.

:mod:`mptb.realdata.slices` runs the baselines (RMS energy VAD, Smart Turn v2, VAP, Kyutai Semantic-VAD)
on either slice, sweeps each system's threshold and selects the operating point *in-slice* (there is no
dev/test split, so these numbers are optimistic). Neither corpus is redistributed: TurnBench dev is
gated and non-commercial, and only aggregate metrics are written by this code.
"""
