"""TRACE-3D: Trajectory & Risk-Aware Clinical Evaluation in 3D.

A 3D-grounded, closed-loop LLM-agent medical evaluation benchmark. An agent reads
a CT-segmentation-derived 3D anatomical scene graph and must turn volumetric
understanding into a safe clinical ACTION (an image-guided biopsy needle
trajectory), scored mostly deterministically against segmentation geometry and
organized by the six-stage STAGER framework (Survey, Triage, Assess, Govern,
Execute, Reflect).

The offline core runs with no network, GPU, or LLM.
"""

__version__ = "0.1.0"
