"""
Main server entry point - runs both REST and gRPC servers
"""

import logging
import os
import signal
import sys
import threading
from typing import Optional

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global server references for graceful shutdown
grpc_server = None
shutdown_event = threading.Event()


def run_rest_server(host: str = "0.0.0.0", port: int = 9000):
    """Run the FastAPI REST server"""
    logger.info(f"Starting REST API server on {host}:{port}")

    config = uvicorn.Config(
        "app.main:app", host=host, port=port, log_level="info", access_log=True
    )
    server = uvicorn.Server(config)
    server.run()


def run_grpc_server(port: int = 50051):
    """Run the gRPC server"""
    global grpc_server

    try:
        from .grpc_server import serve_grpc

        logger.info(f"Starting gRPC server on port {port}")
        grpc_server = serve_grpc(port=port)

        # Wait for shutdown signal
        shutdown_event.wait()

        logger.info("Stopping gRPC server...")
        grpc_server.stop(grace=5)

    except Exception as e:
        logger.error(f"gRPC server error: {str(e)}", exc_info=True)


def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()

    if grpc_server:
        grpc_server.stop(grace=5)

    sys.exit(0)


def main():
    """Main entry point - starts both REST and gRPC servers"""

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration
    rest_port = int(os.getenv("REST_PORT", "9000"))
    grpc_port = int(os.getenv("GRPC_PORT", "50051"))
    enable_grpc = os.getenv("ENABLE_GRPC", "true").lower() in ("true", "1", "yes")

    logger.info("=" * 60)
    logger.info("WhisperX ASR Service - Multi-Protocol Server")
    logger.info("=" * 60)
    logger.info(f"REST API Port: {rest_port}")
    logger.info(f"gRPC Port: {grpc_port} (enabled: {enable_grpc})")
    logger.info(f"Device: {os.getenv('DEVICE', 'cuda')}")
    logger.info(f"Model: {os.getenv('PRELOAD_MODEL', 'large-v3')}")
    logger.info("=" * 60)

    if enable_grpc:
        # Start gRPC server in a separate thread
        grpc_thread = threading.Thread(
            target=run_grpc_server, args=(grpc_port,), daemon=True
        )
        grpc_thread.start()
        logger.info("gRPC server thread started")
    else:
        logger.info("gRPC server disabled by configuration")

    # Run REST server in main thread (blocks until shutdown)
    try:
        run_rest_server(port=rest_port)
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        shutdown_event.set()
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        shutdown_event.set()
        raise


if __name__ == "__main__":
    main()
