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


xs = U17_FWHM_Y.columns.to_numpy(dtype=float)
ys = U17_FWHM_Y.index.to_numpy(dtype=float)
print(U17_FWHM_Y)
XX, YY = np.meshgrid(xs, ys)

params = [
    {"m4.rx": float(x), "m4.ty": float(y)} for x, y in zip(XX.ravel(), YY.ravel(), strict=True)
]
preds = [x["beam_focuser"] for x in agent.ax_client.predict(points=params)]
mean = np.array([x[0] for x in preds]).reshape(XX.shape)
sem = np.array([x[1] for x in preds]).reshape(XX.shape)

pred_df = pd.DataFrame(mean, index=ys, columns=xs)
err = pred_df - U17_FWHM_Y
abs_err = np.abs(err.values)

fig = make_subplots(
    rows=2,
    cols=4,
    specs=[
        [{"colspan": 4}, None, None, None],
        [{}, {}, {}, {}],
    ],
    subplot_titles=[
        "Ax objective view",
        "Truth",
        "Ax prediction of 'Truth' points",
        f"Absolute Error (RMSE={np.sqrt((err.values**2).mean()):.4g})",
        "Predicted SEM",
    ],
    vertical_spacing=0.12,
)

# Top row: reuse traces from existing Plotly figure
for tr in main_figure.data:
    fig.add_trace(tr, row=1, col=1)

vmin = float(U17_FWHM_Y.values.min())
vmax = float(U17_FWHM_Y.values.max())

fig.add_trace(
    go.Heatmap(z=U17_FWHM_Y.values, x=xs, y=ys, zmin=vmin, zmax=vmax, colorscale="Viridis"),
    row=2,
    col=1,
)
fig.add_trace(
    go.Heatmap(z=mean, x=xs, y=ys, zmin=vmin, zmax=vmax, colorscale="Viridis"), row=2, col=2
)
fig.add_trace(
    go.Heatmap(z=abs_err, x=xs, y=ys, zmin=0, zmax=float(abs_err.max()), colorscale="Magma"),
    row=2,
    col=3,
)
fig.add_trace(go.Heatmap(z=sem, x=xs, y=ys, colorscale="Magma"), row=2, col=4)

# Sampled trials overlay on prediction panel
df = agent.ax_client.summarize()
if {"m4.rx", "m4.ty"} <= set(df.columns):
    fig.add_trace(
        go.Scatter(
            x=df["m4.rx"],
            y=df["m4.ty"],
            mode="markers",
            marker={"color": "white", "line": {"color": "black", "width": 1}, "size": 6},
            name="Sampled",
        ),
        row=2,
        col=2,
    )

fig.update_xaxes(title_text="m4.rx", row=2, col=1)
fig.update_xaxes(title_text="m4.rx", row=2, col=2)
fig.update_xaxes(title_text="m4.rx", row=2, col=3)
fig.update_xaxes(title_text="m4.rx", row=2, col=4)
fig.update_yaxes(title_text="m4.ty", row=2, col=1)
fig.update_yaxes(title_text="m4.ty", row=2, col=2)
fig.update_yaxes(title_text="m4.ty", row=2, col=3)
fig.update_yaxes(title_text="m4.ty", row=2, col=4)

fig.update_layout(height=900, width=1400, template="plotly_white")
fig.show()
