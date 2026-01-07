"""
Main server entry point - runs both REST and gRPC servers
"""

import logging
import multiprocessing
import os
import signal
import sys
from typing import Optional

import uvicorn

from .config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global references for graceful shutdown
grpc_process = None


def run_rest_server(host: str = "0.0.0.0", port: Optional[int] = None):
    """Run the FastAPI REST server."""
    port = port or config.rest_port
    logger.info(f"Starting REST API server on {host}:{port}")

    uvi_config = uvicorn.Config(
        "app.main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(uvi_config)
    server.run()


def run_grpc_server(port: Optional[int] = None):
    """Run the gRPC server in a separate process."""
    try:
        from app.grpc_server import serve_grpc

        port = port or config.grpc_port
        logger.info(f"gRPC server process starting on port {port}")
        grpc_server = serve_grpc(port=port)
        grpc_server.wait_for_termination()

    except Exception as e:
        logger.error(f"gRPC server error: {e}", exc_info=True)
        sys.exit(1)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")

    if grpc_process and grpc_process.is_alive():
        logger.info("Terminating gRPC server process...")
        grpc_process.terminate()
        grpc_process.join(timeout=5)
        if grpc_process.is_alive():
            logger.warning("Force killing gRPC process...")
            grpc_process.kill()

    sys.exit(0)


def main():
    """Main entry point - starts both REST and gRPC servers."""
    global grpc_process

    # Set multiprocessing start method (important for CUDA)
    multiprocessing.set_start_method("spawn", force=True)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration
    enable_grpc = os.getenv("ENABLE_GRPC", "true").lower() in ("true", "1", "yes")

    logger.info("=" * 60)
    logger.info("WhisperX ASR Service - Multi-Protocol Server")
    logger.info("=" * 60)
    logger.info(f"REST API Port: {config.rest_port}")
    logger.info(f"gRPC Port: {config.grpc_port} (enabled: {enable_grpc})")
    logger.info(f"Device: {config.device}")
    logger.info(f"Model: {config.default_model}")
    logger.info(f"Diarization Model: {config.diarization_model}")
    logger.info("=" * 60)

    if enable_grpc:
        # Start gRPC server in a separate process
        grpc_process = multiprocessing.Process(
            target=run_grpc_server,
            args=(config.grpc_port,),
            name="gRPC-Server",
        )
        grpc_process.start()
        logger.info(f"gRPC server process started (PID: {grpc_process.pid})")
    else:
        logger.info("gRPC server disabled by configuration")

    # Run REST server in main process (blocks until shutdown)
    try:
        run_rest_server(port=config.rest_port)
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise
    finally:
        # Cleanup gRPC process on exit
        if grpc_process and grpc_process.is_alive():
            logger.info("Stopping gRPC server process...")
            grpc_process.terminate()
            grpc_process.join(timeout=5)
            if grpc_process.is_alive():
                grpc_process.kill()


if __name__ == "__main__":
    main()
