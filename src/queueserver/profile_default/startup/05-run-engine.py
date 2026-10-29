print("Loading run engine...")
from bluesky import RunEngine  # noqa
from bluesky.callbacks.zmq import Publisher  # noqa

RE = RunEngine()
RE.subscribe(Publisher("tcp://127.0.0.1:5577"))
