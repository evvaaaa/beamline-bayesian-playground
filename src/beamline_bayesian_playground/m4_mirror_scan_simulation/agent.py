import argparse
from datetime import datetime
from pathlib import Path

from ax.adapter.registry import Generators
from ax.api.client import Client
from ax.api.utils.generation_strategy_dispatch import choose_generation_strategy
from ax.api.utils.structs import GenerationStrategyDispatchStruct
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import AutoTransitionAfterGen
from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_adaptive.ax import Objective, RangeDOF
from bluesky_adaptive.ax.queueserver_agent import QueueserverAgent
from bluesky_queueserver_api.zmq import REManagerAPI
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from tiled.client import from_uri

from beamline_bayesian_playground.m4_mirror_scan_simulation.evaluator import BeamFocusEvaluator

TY_BOUNDS = (-3000, -2800)
RX_BOUNDS = (-2020, -1820)
EXPLORATION_BETA = 50.0
INITIALIZATION_TRIALS = 16


def configure_exploration(client: Client) -> None:
    """Combine space-filling measurements with uncertainty-weighted minimization."""
    initial_strategy = choose_generation_strategy(
        struct=GenerationStrategyDispatchStruct(initialization_budget=INITIALIZATION_TRIALS),
        botorch_acqf_class=qUpperConfidenceBound,
    )
    model_node = initial_strategy.nodes_by_name["MBM"]
    model_node.generator_specs[0].generator_kwargs["botorch_acqf_options"] = {
        "beta": EXPLORATION_BETA
    }
    minimization_node = GenerationNode(
        name="MBM",
        generator_specs=model_node.generator_specs,
        should_deduplicate=True,
        suggested_experiment_status=model_node.suggested_experiment_status,
        transition_criteria=[
            AutoTransitionAfterGen("Explore1", continue_trial_generation=False)
        ],
    )
    explore_nodes = [
        GenerationNode(
            name=name,
            generator_specs=[GeneratorSpec(generator_enum=Generators.SOBOL)],
            should_deduplicate=True,
            transition_criteria=[
                AutoTransitionAfterGen(next_name, continue_trial_generation=False)
            ],
        )
        for name, next_name in (("Explore1", "Explore2"), ("Explore2", "MBM"))
    ]
    strategy = GenerationStrategy(
        name="M4MappingAndMinimization",
        nodes=[
            *(
                minimization_node if node.name == "MBM" else node
                for node in initial_strategy.nodes_by_name.values()
            ),
            *explore_nodes,
        ],
    )
    client.set_generation_strategy(strategy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--acquisition",
        choices=("explore", "default"),
        default="explore",
        help="Use space-filling and uncertainty-weighted minimization (default), or Ax's strategy",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    re_manager_api = REManagerAPI(zmq_control_addr="tcp://127.0.0.1:60615")
    tiled_client = from_uri("http://127.0.0.1:8000", api_key="secret")
    document_dispatcher = RemoteDispatcher(("127.0.0.1", 5578))
    if not re_manager_api.status()["worker_environment_exists"]:  # type: ignore
        re_manager_api.environment_open()
    re_manager_api.wait_for_idle()

    evaluator = BeamFocusEvaluator(tiled_client)
    agent = QueueserverAgent(
        re_manager_api=re_manager_api,
        document_dispatcher=document_dispatcher,
        sensors=["mocked_bpm_ue48_12_11_26"],
        dofs=[
            RangeDOF(actuator="m4_ty", bounds=TY_BOUNDS, parameter_type="float"),
            RangeDOF(actuator="m4_rx", bounds=RX_BOUNDS, parameter_type="float"),
        ],
        objectives=[Objective(name=evaluator.OUTCOME_NAME, minimize=True)],
        evaluation_function=evaluator,
        acquisition_plan="default_acquire",
    )
    if args.acquisition == "explore":
        configure_exploration(agent.ax_client)

    future = agent.run(iterations=args.iterations, n_points=1)
    future.result()

    print(agent.ax_client.summarize())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent.ax_client.save_to_json_file(
        str(
            args.output_dir
            / f"{args.iterations}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        )
    )


if __name__ == "__main__":
    main()
