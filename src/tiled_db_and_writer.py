"""Starts the tmp tiled db on one thread, and the writer over zmq on another."""

import os

from bluesky.callbacks.zmq import RemoteDispatcher
from bluesky_tiled_plugins import TiledWriter
from tiled.client import from_uri
from tiled.server import SimpleTiledServer

ZMQ_URI = os.environ["ZMQ_URI"]
TILED_API_KEY = os.environ["TILED_API_KEY"]
TILED_URI = os.environ["TILED_URI"]

# Initialize the Tiled server and client
tiled_server = SimpleTiledServer(port=int(TILED_URI.split(":")[-1]), api_key=TILED_API_KEY)
print(f"Started tiled server at {tiled_server.uri}")
tiled_client = from_uri(tiled_server.uri)


# The tiled interface is really quite annoying. There doesn't seem to
# be a convenient way to quickly start a test db.
writer = TiledWriter(tiled_client)
dispatcher = RemoteDispatcher(ZMQ_URI)
dispatcher.subscribe(writer)
print(f"Staring zmq dispatcher at {ZMQ_URI}...")
dispatcher.start()
