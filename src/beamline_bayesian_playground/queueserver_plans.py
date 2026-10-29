"""All plans here are imported in the queueserver established in docker compose."""

from collections.abc import Mapping, Sequence
from typing import Any

import bluesky_adaptive.plans as bap
from bluesky.plan_stubs import mov, mv, one_shot, rel_set  # noqa
from bluesky.plans import PerStep, count, grid_scan  # noqa
from bluesky.protocols import Readable
from bluesky.utils import MsgGenerator, plan
from bluesky_adaptive.plans import *  # noqa
from bluesky_adaptive.protocols import (
    Actuator,
)


@plan
def default_acquire(
    suggestions: Sequence[Mapping],
    actuators: Sequence[Actuator],  # type: ignore
    sensors: Sequence[Readable] | None = None,
    md: Mapping[str, Any] | None = None,
    *,
    per_step: PerStep | None = None,
    **kwargs: Any,
) -> MsgGenerator[str]:
    # The exact same as bluesky_adaptive.plans.default_acquire, but
    # * Without the EventCollectable protocol hint. It seems the queueserver has some
    #   problems with hinting EventCollectable.
    # * Using the name of the suggetions instead of the . access queueserver, e.g m4.rx.
    #   This seems like a bug in bluesky_adaptive.
    patched_suggestions = []
    for suggestion in suggestions:
        patched_suggestion = {}
        for key, value in suggestion.items():
            patched_suggestion[key.replace(".", "-")] = value
        patched_suggestions.append(patched_suggestion)
    md["blop_suggestions"] = patched_suggestions

    ret = yield from bap.default_acquire(
        patched_suggestions, actuators, sensors=sensors, md=md, per_step=per_step, **kwargs
    )
    return ret
