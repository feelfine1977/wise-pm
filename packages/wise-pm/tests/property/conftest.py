"""Hypothesis profiles for the property-based suite.

``HYPOTHESIS_PROFILE=ci-long`` selects the long profile (300 examples per
property); the default runs 40 examples so the suite stays fast locally.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "default",
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile("ci-long", parent=settings.get_profile("default"), max_examples=300)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))
