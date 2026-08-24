"""Agent4LR: Localization-Refinement phase of FlexFL.

Takes the top-20 candidate list produced by Agent4SR and refines it to a
top-5 via a narrower LLM ReAct loop. Behavioural reference:
``~/git/FlexFL/FlexFL/src/modular_pipeline.py``.
"""
