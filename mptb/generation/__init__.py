"""MultiPartyTurnBench data-generation pipeline (paper Section 3.1, Figure 2, Appendix C).

Modules, in pipeline order:

* :mod:`~mptb.generation.strata`  -- interaction axes and the stratified conversation-spec sampler.
* :mod:`~mptb.generation.split`   -- voice-disjoint dev/test partition of the voice pool.
* :mod:`~mptb.generation.llm`     -- OpenAI-compatible chat client used by Stage 1.
* :mod:`~mptb.generation.stage1_script`     -- Stage 1: event-conditioned LLM script generation.
* :mod:`~mptb.generation.stage2_orchestrate` -- Stage 2: floor-state orchestration -> timeline + gold events.
* :mod:`~mptb.generation.tts`      -- Stage 3: Kokoro-82M speech synthesis.
* :mod:`~mptb.generation.assemble` -- Stage 3/5: per-speaker channel assembly.
* :mod:`~mptb.generation.noise`    -- Stage 4: acoustic scene (ambient bed, MUSAN events, distant speech).
* :mod:`~mptb.generation.qc`       -- acoustic quality control of the clean channels.
* :mod:`~mptb.generation.pipeline` -- end-to-end, checkpointed generation of one corpus.
* :mod:`~mptb.generation.checks`   -- corpus-level consistency checks (oracle round trip, QC, balance, split).
* :mod:`~mptb.generation.stats`    -- corpus statistics (``metadata/dataset_stats.json``, Appendix D tables).

Heavy dependencies (numpy, soundfile, torch/kokoro, openai) are imported lazily or only by the
modules that need them; :mod:`strata`, :mod:`split`, :mod:`stage1_script`, :mod:`stage2_orchestrate`
and :mod:`stats` are standard-library only.
"""
