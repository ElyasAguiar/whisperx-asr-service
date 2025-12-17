# gRPC API Guide

This service now supports gRPC in addition to the REST API, making it compatible with NVIDIA Riva and other gRPC-based ASR clients.

## Overview

The gRPC API follows the NVIDIA Riva ASR protocol, providing:
- **Streaming recognition** for real-time audio processing
- **Batch recognition** for audio files
- **Speaker diarization** support
- **Word-level timestamps** with confidence scores
- **Model-agnostic architecture** - easily swap between different ASR models

## Quick Start

### 1. Enable gRPC (Default: Enabled)

gRPC is enabled by default. To configure, set in `.env`:

```bash
ENABLE_GRPC=true          # Enable/disable gRPC server
GRPC_PORT=50051           # gRPC port (default: 50051)
REST_PORT=9000            # REST port (default: 9000)
```

### 2. Start the Service

```bash
# With docker-compose (both REST and gRPC enabled)
docker compose up -d

# Check logs
docker compose logs -f
```

The service will start:
- **REST API** on port `9000`
- **gRPC API** on port `50051`

### 3. Test with Example Client

```bash
# Install dependencies
pip install grpcio grpcio-tools

# Run example client
python3 examples/grpc_client_example.py /path/to/audio.wav \
    --host localhost \
    --port 50051 \
    --language en \
    --model large-v3
```

## gRPC API Reference

### Service Definition

```protobuf
service AsrService {
  // Streaming recognition for real-time audio
  rpc StreamingRecognize(stream StreamingRecognizeRequest) 
      returns (stream StreamingRecognizeResponse);
  
  // Single request recognition for audio files
  rpc Recognize(RecognizeRequest) returns (RecognizeResponse);
}
```

### Recognize (Batch Recognition)

Process a complete audio file in a single request.

**Request:**

```python
import grpc
from app.grpc_generated import asr_pb2, asr_pb2_grpc

# Create channel
channel = grpc.insecure_channel('localhost:50051')
stub = asr_pb2_grpc.AsrServiceStub(channel)

# Read audio file
with open('audio.wav', 'rb') as f:
    audio_content = f.read()

# Build request
config = asr_pb2.RecognitionConfig(
    encoding=asr_pb2.RecognitionConfig.LINEAR_PCM,
    sample_rate_hertz=16000,
    language_code="en",
    model="large-v3",
    enable_word_time_offsets=True,
    enable_speaker_diarization=True,
    diarization_speaker_count=2  # Optional: exact speaker count
)

audio = asr_pb2.RecognitionAudio(content=audio_content)
request = asr_pb2.RecognizeRequest(config=config, audio=audio)

# Make request
response = stub.Recognize(request)

# Process results
for result in response.results:
    print(f"Language: {result.language_code}")
    print(f"Duration: {result.audio_processed}s")
    
    for alternative in result.alternatives:
        print(f"Transcript: {alternative.transcript}")
        print(f"Confidence: {alternative.confidence}")
        
        for word in alternative.words:
            print(f"{word.word}: {word.start_time}-{word.end_time} "
                  f"(speaker={word.speaker_tag})")
```

**Response:**

```json
{
  "results": [
    {
      "alternatives": [
        {
          "transcript": "Hello, welcome to the meeting.",
          "confidence": 1.0,
          "words": [
            {
              "word": "Hello",
              "start_time": 0.5,
              "end_time": 0.8,
              "confidence": 0.95,
              "speaker_tag": 0
            },
            ...
          ]
        }
      ],
      "language_code": "en",
      "audio_processed": 120.5
    }
  ]
}
```

### StreamingRecognize (Streaming Recognition)

Process audio in chunks for real-time transcription.

**Request:**

```python
def request_generator():
    # First request: configuration
    config = asr_pb2.RecognitionConfig(
        encoding=asr_pb2.RecognitionConfig.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code="en",
        model="large-v3",
        enable_word_time_offsets=True
    )
    
    streaming_config = asr_pb2.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )
    
    yield asr_pb2.StreamingRecognizeRequest(
        streaming_config=streaming_config
    )
    
    # Subsequent requests: audio chunks
    with open('audio.wav', 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            yield asr_pb2.StreamingRecognizeRequest(
                audio_content=chunk
            )

# Send streaming request
responses = stub.StreamingRecognize(request_generator())

# Process streaming responses
for response in responses:
    for result in response.results:
        if result.is_final:
            print(f"Final: {result.alternatives[0].transcript}")
        else:
            print(f"Interim: {result.alternatives[0].transcript}")
```

## Configuration Options

### RecognitionConfig Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `encoding` | enum | Audio encoding format | LINEAR_PCM |
| `sample_rate_hertz` | int | Sample rate in Hz | 16000 |
| `audio_channel_count` | int | Number of audio channels | 1 |
| `language_code` | string | Language code (e.g., "en", "es") | Auto-detect |
| `model` | string | Model name (e.g., "large-v3") | large-v3 |
| `enable_word_time_offsets` | bool | Return word-level timestamps | true |
| `enable_speaker_diarization` | bool | Enable speaker diarization | false |
| `diarization_speaker_count` | int | Exact number of speakers | Auto |
| `min_speaker_count` | int | Minimum speakers | Auto |
| `max_speaker_count` | int | Maximum speakers | Auto |
| `task` | string | "transcribe" or "translate" | transcribe |
| `initial_prompt` | string | Initial prompt for model | - |

### Supported Audio Encodings

- `LINEAR_PCM` - Linear PCM (WAV)
- `FLAC` - FLAC format
- `MP3` - MP3 format
- `OGG_OPUS` - Opus in Ogg container
- `MULAW` - μ-law encoding
- `ALAW` - A-law encoding

## Advanced Usage

### Speaker Diarization

Identify different speakers in audio:

```python
config = asr_pb2.RecognitionConfig(
    # ... other config ...
    enable_speaker_diarization=True,
    diarization_speaker_count=3  # If you know exact number
    # OR
    # min_speaker_count=2,
    # max_speaker_count=5
)
```

The response will include speaker tags in word-level results:

```python
for word in alternative.words:
    speaker = f"SPEAKER_{word.speaker_tag:02d}"
    print(f"{speaker}: {word.word}")
```

### Multi-Language Support

The service supports 90+ languages. Specify language code:

```python
config = asr_pb2.RecognitionConfig(
    language_code="es",  # Spanish
    # or "fr" (French), "de" (German), "pt" (Portuguese), etc.
)
```

Leave empty or set to `None` for automatic language detection.

### Translation

Translate audio to English while transcribing:

```python
config = asr_pb2.RecognitionConfig(
    language_code="es",  # Source language
    task="translate"     # Translate to English
)
```

## Integration with NVIDIA Riva

This service implements the same gRPC protocol as NVIDIA Riva ASR, making it compatible with Riva clients and applications.

### Replacing Riva ASR

If you have an application using Riva ASR, you can point it to this service:

1. Update the gRPC endpoint:
   ```python
   # Old Riva endpoint
   channel = grpc.insecure_channel('riva-server:50051')
   
   # New WhisperX endpoint
   channel = grpc.insecure_channel('whisperx-server:50051')
   ```

2. Use the same Riva proto definitions (they're compatible)

3. No code changes needed - the API is compatible!

### Differences from Riva

While compatible, there are some differences:
- WhisperX uses different underlying models (OpenAI Whisper)
- Some advanced Riva features may not be implemented
- Performance characteristics differ

## Model-Agnostic Architecture

The service is designed with an abstraction layer that makes it easy to swap ASR models:

```python
from app.asr_interface import ASRInterface, ASRConfig, ASRResult

class CustomASR(ASRInterface):
    def transcribe(self, audio_data: bytes, config: ASRConfig) -> ASRResult:
        # Your custom ASR implementation
        pass
```

This allows you to use different models without changing the gRPC API:
- OpenAI Whisper (current)
- NVIDIA Riva
- Google Speech-to-Text
- Custom models

## Performance Considerations

### GPU Memory

gRPC requests use the same GPU resources as REST API:
- Monitor with `nvidia-smi`
- Adjust `BATCH_SIZE` and `MAX_FILE_SIZE_MB` as needed
- Use smaller models for lower VRAM requirements

### Concurrency

The gRPC server uses a thread pool:
- Default: 10 worker threads
- Configure via environment: `GRPC_MAX_WORKERS=20`
- Each request holds GPU during processing

### Streaming vs Batch

- **Batch (`Recognize`)**: Better for files, single GPU allocation
- **Streaming (`StreamingRecognize`)**: Real-time, but holds resources longer

## Troubleshooting

### gRPC Server Not Starting

Check logs:
```bash
docker compose logs whisperx-asr | grep gRPC
```

Verify port is not in use:
```bash
netstat -tuln | grep 50051
```

### Connection Refused

Ensure gRPC is enabled:
```bash
# In .env
ENABLE_GRPC=true
```

Check firewall rules:
```bash
# Allow port 50051
sudo ufw allow 50051/tcp
```

### Model Not Found

Preload models on startup:
```bash
# In .env
PRELOAD_MODEL=large-v3
```

### Out of Memory

Reduce batch size or use smaller model:
```bash
BATCH_SIZE=8
PRELOAD_MODEL=medium
```

## Examples

### Python Client

See `examples/grpc_client_example.py` for a complete Python client implementation.

### Go Client

```go
// Generate Go code from proto
// protoc --go_out=. --go-grpc_out=. proto/asr.proto

package main

import (
    "context"
    "io/ioutil"
    "log"
    
    "google.golang.org/grpc"
    pb "your-module/asr"
)

func main() {
    conn, err := grpc.Dial("localhost:50051", grpc.WithInsecure())
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    
    client := pb.NewAsrServiceClient(conn)
    
    audio, _ := ioutil.ReadFile("audio.wav")
    
    resp, err := client.Recognize(context.Background(), &pb.RecognizeRequest{
        Config: &pb.RecognitionConfig{
            Encoding:              pb.RecognitionConfig_LINEAR_PCM,
            SampleRateHertz:       16000,
            LanguageCode:          "en",
            Model:                 "large-v3",
            EnableWordTimeOffsets: true,
        },
        Audio: &pb.RecognitionAudio{
            AudioSource: &pb.RecognitionAudio_Content{
                Content: audio,
            },
        },
    })
    
    if err != nil {
        log.Fatal(err)
    }
    
    for _, result := range resp.Results {
        log.Printf("Transcript: %s", result.Alternatives[0].Transcript)
    }
}
```

### Node.js Client

```javascript
const grpc = require('@grpc/grpc-js');
const protoLoader = require('@grpc/proto-loader');
const fs = require('fs');

// Load proto
const packageDefinition = protoLoader.loadSync('proto/asr.proto');
const asr = grpc.loadPackageDefinition(packageDefinition).whisperx.asr;

// Create client
const client = new asr.AsrService(
    'localhost:50051',
    grpc.credentials.createInsecure()
);

// Read audio
const audio = fs.readFileSync('audio.wav');

// Make request
client.Recognize({
    config: {
        encoding: 'LINEAR_PCM',
        sample_rate_hertz: 16000,
        language_code: 'en',
        model: 'large-v3',
        enable_word_time_offsets: true
    },
    audio: {
        content: audio
    }
}, (err, response) => {
    if (err) {
        console.error(err);
        return;
    }
    
    response.results.forEach(result => {
        result.alternatives.forEach(alt => {
            console.log('Transcript:', alt.transcript);
        });
    });
});
```

## Security Considerations

⚠️ **Important Security Notes:**

- The gRPC server uses **insecure channels** (no TLS/SSL)
- There is **no authentication or authorization**
- Suitable for **internal networks only**

For production:
- Use TLS/SSL certificates
- Implement authentication (API keys, mTLS)
- Deploy behind a secure gateway
- Use firewall rules to restrict access

## Further Reading

- [gRPC Documentation](https://grpc.io/docs/)
- [NVIDIA Riva ASR](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html)
- [Protocol Buffers](https://protobuf.dev/)
- [WhisperX Repository](https://github.com/m-bain/whisperX)
