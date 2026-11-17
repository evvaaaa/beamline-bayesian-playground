import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from ax.analysis.plotly.plotly_analysis import PlotlyAnalysisCard
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_adaptive.ax import Objective, RangeDOF
from bluesky_adaptive.ax.queueserver_agent import QueueserverAgent
from bluesky_queueserver_api.zmq import REManagerAPI
from tiled.client import from_uri
from beamline_bayesian_playground.m4_mirror_scan_simulation.evaluator import BeamFocusEvaluator

# ====== Connection to docker-compose environment.
RE_MANAGER_API = REManagerAPI(zmq_control_addr="tcp://127.0.0.1:60615")
TILED_CLIENT = from_uri("http://127.0.0.1:8000", api_key="secret")
DOCUMENT_DISPATCHER = RemoteDispatcher(("127.0.0.1", 5578))
if not RE_MANAGER_API.status()["worker_environment_exists"]:  # type: ignore
    RE_MANAGER_API.environment_open()
RE_MANAGER_API.wait_for_idle()

# ====== Connection to docker-compose environment.
EVALUATOR = BeamFocusEvaluator(TILED_CLIENT)
TY_BOUNDS = (-3000, -2800)
RX_BOUNDS = (-2020, -1820)
OBJECTIVES = [
    Objective(name=EVALUATOR.OUTCOME_NAME, minimize=False),
]
SENSORS = ["mocked_bpm_ue48_12_11_26"]
DEGREES_OF_FREEDOM = [
    RangeDOF(actuator="m4_ty", bounds=TY_BOUNDS, parameter_type="float"),
    RangeDOF(actuator="m4_rx", bounds=RX_BOUNDS, parameter_type="float"),
]

agent = QueueserverAgent(
    re_manager_api=RE_MANAGER_API,
    document_dispatcher=DOCUMENT_DISPATCHER,
    sensors=SENSORS,
    dofs=DEGREES_OF_FREEDOM,
    objectives=OBJECTIVES,
    evaluation_function=EVALUATOR,
    acquisition_plan="default_acquire",
)

# ======= Start the agent.
future = agent.run(iterations=60, n_points=1)
future.result()  # block until completion


# ======= Summarize and plot the run along with error of the prediction against real values and SEM.
print(agent.ax_client.summarize())
figure = agent.plot_objective(
    x_dof_name="m4.rx", y_dof_name="m4.ty", objective_name="beam_focuser"
)[0]


if isinstance(figure, PlotlyAnalysisCard):
    main_figure = figure.get_figure()
else:
    raise RuntimeError(f"A valid plot was not generated: {figure}")
