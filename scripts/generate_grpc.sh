#!/bin/bash
# Generate gRPC Python code from proto files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROTO_DIR="$PROJECT_ROOT/proto"
OUTPUT_DIR="$PROJECT_ROOT/app/grpc_generated"

echo "Generating gRPC code from proto files..."
echo "Proto directory: $PROTO_DIR"
echo "Output directory: $OUTPUT_DIR"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Generate Python code from proto files
python3 -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUTPUT_DIR" \
    --grpc_python_out="$OUTPUT_DIR" \
    --pyi_out="$OUTPUT_DIR" \
    "$PROTO_DIR"/*.proto

# Create __init__.py to make it a package
touch "$OUTPUT_DIR/__init__.py"

echo "✓ gRPC code generation complete!"
echo "Generated files in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
