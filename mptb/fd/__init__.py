"""Full-duplex (FD) behavioural evaluation of voice agents (paper Section 5, RQ4; Table 4; Appendices F-H).

The experiment turns 120 gold MPTB events into interactive floor-control probes and records whether a
full-duplex agent, joining the conversation as an additional party, *starts* speaking after a genuine
end of turn (and not during a mid-turn hold pause) and *stops* speaking when it is interrupted (and not
when it hears a backchannel).

Pipeline (each step has a CLI under ``scripts/fd/``):

1. :mod:`mptb.fd.scenarios`   select the 120 scenarios from the released gold annotations.
2. :mod:`mptb.fd.audio_packs` cut the audio streamed to the agents (one pack per scenario).
3. :mod:`mptb.fd.runner`      run every scenario x agent x repeat through :mod:`mptb.fd.protocols`
                              using a provider adapter from :mod:`mptb.fd.adapters`.
4. :mod:`mptb.fd.offline`     the same agents as offline detectors on the same events (Appendix G).
5. :mod:`mptb.fd.analysis`    per-trial table, per-agent metrics, bootstrap CIs, offline-online association.

"""
