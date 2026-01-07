"""
Main server entry point - runs REST API server
"""

import logging
import os

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point - starts REST API server"""
    # Get configuration
    rest_port = int(os.getenv("REST_PORT", "9000"))

    logger.info("=" * 60)
    logger.info("WhisperX ASR Service")
    logger.info("=" * 60)
    logger.info(f"REST API Port: {rest_port}")
    logger.info(f"Device: {os.getenv('DEVICE', 'cuda')}")
    logger.info(f"Model: {os.getenv('PRELOAD_MODEL', 'large-v3')}")
    logger.info("=" * 60)

    # Run REST server
    try:
        config = uvicorn.Config(
            "app.main:app",
            host="0.0.0.0",
            port=rest_port,
            log_level="info",
            access_log=True,
        )
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Server error: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
