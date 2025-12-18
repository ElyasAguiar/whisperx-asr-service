# Tests

Unit tests for the gRPC implementation and ASR abstraction layer.

## Running Tests

```bash
# From project root
python3 tests/test_grpc_integration.py
```

## What's Tested

- **gRPC Protocol Structure**: Verifies proto definitions and message structures
- **ASR Interface**: Tests the abstract base class and data structures
- **Model Abstraction**: Validates configuration and result formats
- **gRPC Server Components**: Tests servicer creation and conversion logic

## Note

These tests focus on structure and interfaces, not runtime behavior. Full integration tests require:
- GPU environment
- WhisperX and dependencies installed
- Running gRPC server

For full integration testing, see the `examples/` directory for client examples.
