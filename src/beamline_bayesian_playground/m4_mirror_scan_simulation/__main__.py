"""Script to be ran from outside the queueserver."""

import time
from collections.abc import Mapping, Sequence

from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_adaptive.ax import Objective, RangeDOF
from bluesky_adaptive.ax.queueserver_agent import QueueserverAgent
from bluesky_queueserver_api.zmq import REManagerAPI
from tiled.client import from_uri
from tiled.client.container import Container

# ====== Bluesky adaptive specification of problem and optimizers.

TY_BOUNDS = (200.0, 700.0)
RX_BOUNDS = (50.0, 250.0)
OUTCOME_NAME = "beam_focuser"
OBJECTIVES = [
    Objective(name=OUTCOME_NAME, minimize=False),
]
SENSORS = ["u17_mocked_bpm"]
DEGREES_OF_FREEDOM = [
    RangeDOF(actuator="m4.ty", bounds=TY_BOUNDS, parameter_type="float"),
    RangeDOF(actuator="m4.rx", bounds=RX_BOUNDS, parameter_type="float"),
]


class BeamFocusEvaluator:
    """Reads detector data from Tiled and computes the beam objective.

    The beam objective is to get a high full width half maximum in
    """

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
                    OUTCOME_NAME: self.fwhm_evaulation(fwhm_x[idx], fwhm_y[idx]),
                }
            )

        return outcomes


if __name__ == "__main__":
    # ====== Connection to docker-compose environment.
    RE_MANAGER_API = REManagerAPI(zmq_control_addr="tcp://127.0.0.1:60615")
    TILED_CLIENT = from_uri("http://127.0.0.1:8000", api_key="secret")
    DOCUMENT_DISPATCHER = RemoteDispatcher(("127.0.0.1", 5578))
    if not RE_MANAGER_API.status()["worker_environment_exists"]:
        RE_MANAGER_API.environment_open()
    RE_MANAGER_API.wait_for_idle()

    # ====== Agent instantiation.
    agent = QueueserverAgent(
        re_manager_api=RE_MANAGER_API,
        document_dispatcher=DOCUMENT_DISPATCHER,
        sensors=SENSORS,
        dofs=DEGREES_OF_FREEDOM,
        objectives=OBJECTIVES,
        evaluation_function=BeamFocusEvaluator(TILED_CLIENT),
        acquisition_plan="default_acquire",
    )

    # Run the agent.
    future = agent.run(iterations=40, n_points=1)
    future.result()  # block until completion
    print(agent.ax_client.summarize())
    figure = agent.plot_objective(
        x_dof_name="m4.rx", y_dof_name="m4.ty", objective_name="beam_focuser"
    )[0]
    if isinstance(figure, PlotlyAnalysisCard):
        figure.get_figure().show()
