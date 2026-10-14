"""LLM harness for the iSpatialGym eval instances: frontier models drive R1Pro through a fixed, non-learned motion layer
inside ``run_isg --mode policy`` (see ``README.md`` in this package). Simulator-free modules import without Isaac Sim;
``core.embodiment`` / ``core.planning`` / ``core.motion`` need a running simulator (``policy.HarnessPolicy.bind``)."""
