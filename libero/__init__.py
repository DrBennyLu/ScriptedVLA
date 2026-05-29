"""LIBERO dataset training helpers, WebSocket eval clients, and debugging tools."""

from pathlib import Path

from libero._bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

LIBERO_DIR = Path(__file__).resolve().parent
REPO_ROOT = LIBERO_DIR.parent

DEFAULT_CONFIG = "libero/config_libero_object.yaml"
DEFAULT_PRETRAIN_CONFIG = "libero/config_libero_object_pretrain.yaml"
