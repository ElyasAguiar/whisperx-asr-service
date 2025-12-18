# gRPC Client Examples

This directory contains example clients for the gRPC API.

## Prerequisites

Install gRPC Python dependencies:

```bash
pip install grpcio grpcio-tools
```

## Usage

### Basic Usage

Transcribe an audio file using the gRPC API:

```bash
python3 grpc_client_example.py /path/to/audio.wav
```

### With Options

```bash
# Specify server and language
python3 grpc_client_example.py audio.wav \
    --host localhost \
    --port 50051 \
    --language en \
    --model large-v3

# With speaker diarization
python3 grpc_client_example.py audio.wav \
    --language en \
    --num-speakers 2

# Disable diarization
python3 grpc_client_example.py audio.wav \
    --no-diarization

# Use streaming API
python3 grpc_client_example.py audio.wav \
    --streaming
```

## Available Options

- `--host`: gRPC server host (default: localhost)
- `--port`: gRPC server port (default: 50051)
- `--language`: Language code (default: en)
- `--model`: Model name (default: large-v3)
- `--no-diarization`: Disable speaker diarization
- `--num-speakers`: Number of speakers (if known)
- `--streaming`: Use streaming API instead of batch

## Audio Format Requirements

The example assumes LINEAR_PCM (WAV) format. For other formats:

1. Convert to WAV first:
   ```bash
   ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

2. Or modify the example to specify the correct encoding in `RecognitionConfig`

## Creating Your Own Client

See the example code for reference. Basic structure:

```python
import grpc
from app.grpc_generated import asr_pb2, asr_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = asr_pb2_grpc.AsrServiceStub(channel)

# Build request
config = asr_pb2.RecognitionConfig(...)
audio = asr_pb2.RecognitionAudio(content=audio_bytes)
request = asr_pb2.RecognizeRequest(config=config, audio=audio)

# Send request
response = stub.Recognize(request)
```

## Troubleshooting

**Import error when running example:**
```bash
# Make sure you're running from the project root
cd /path/to/whisperx-asr-service
python3 examples/grpc_client_example.py audio.wav
```

**Connection refused:**
```bash
# Check if gRPC server is running
docker compose ps

# Check logs
docker compose logs whisperx-asr
```

**Module not found errors:**
```bash
# Regenerate gRPC code
bash scripts/generate_grpc.sh
```
