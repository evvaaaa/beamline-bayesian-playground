import time
from collections.abc import Mapping, Sequence
from tiled.client.container import Container


class BeamFocusEvaluator:
    """Reads detector data from Tiled and computes the beam objective.

    The beam objective is to get a high full width half maximum in
    """

    OUTCOME_NAME = "beam_focus_evaluator"

    def __init__(self, tiled_client: Container, timeout: float = 30.0, poll_interval: float = 0.5):
        self.tiled_client = tiled_client
        self.timeout = timeout
        self.poll_interval = poll_interval

    def _wait_for_run(self, uid: str) -> Container:
        """Poll Tiled until the run with the given UID is available."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                return self.tiled_client[uid]
            except KeyError:
                time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Run '{uid}' not found in Tiled after {self.timeout}s. "
            "The ZMQ-Tiled bridge may not be running."
        )

    def _wait_for_detector_data(self, run: Container):
        """Poll Tiled until the detector data path is readable."""
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                return run["primary"].read()
            except KeyError:
                time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Run '{run.metadata['start']['uid']}' was not "
            f"readable after {self.timeout}s. The ZMQ-Tiled bridge may still be writing the run."
            ""
        )

    def fwhm_evaulation(self, x: float, y: float) -> float:
        """Check how 'good' the beam focus is."""
        # Right now we just want to maximize y...
        # I'll think about how x should play into this later.
        return y

    def __call__(self, uid: str, suggestions: Sequence[Mapping]) -> Sequence[Mapping]:
        run = self._wait_for_run(uid)

        data = self._wait_for_detector_data(run)
        # Read the detector values from the primary data stream
        fwhm_x = data["bpm-fwhm_x"].to_numpy().tolist()
        fwhm_y = data["bpm-fwhm_y"].to_numpy().tolist()

        acquisition_order: list[str] = run.metadata["start"]["blop_acquisition_order"]  # type: ignore

        outcomes = []
        for idx, suggestion_id in enumerate(acquisition_order):
            outcomes.append(
                {
                    "_id": suggestion_id,
                    self.OUTCOME_NAME: self.fwhm_evaulation(fwhm_x[idx], fwhm_y[idx]),
                }
            )

        return outcomes
