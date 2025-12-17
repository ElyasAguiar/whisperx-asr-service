#!/usr/bin/env python3
"""
Example gRPC client for WhisperX ASR Service

This demonstrates how to use the gRPC API to transcribe audio files.
Compatible with NVIDIA Riva ASR protocol.
"""

import sys
import os
import argparse
import logging

# Add parent directory to path to import generated proto files
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import grpc
from app.grpc_generated import asr_pb2, asr_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recognize_file(
    audio_file_path: str,
    host: str = "localhost",
    port: int = 50051,
    language: str = "en",
    model: str = "large-v3",
    enable_diarization: bool = True,
    num_speakers: int = None
):
    """
    Transcribe an audio file using the gRPC API
    
    Args:
        audio_file_path: Path to audio file
        host: gRPC server host
        port: gRPC server port
        language: Language code (e.g., 'en', 'es', 'fr')
        model: Model name (e.g., 'large-v3', 'medium', 'small')
        enable_diarization: Enable speaker diarization
        num_speakers: Number of speakers (if known)
    """
    # Read audio file
    logger.info(f"Reading audio file: {audio_file_path}")
    with open(audio_file_path, 'rb') as f:
        audio_content = f.read()
    
    logger.info(f"Audio file size: {len(audio_content)} bytes")
    
    # Create gRPC channel
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = asr_pb2_grpc.AsrServiceStub(channel)
    
    # Build request
    config = asr_pb2.RecognitionConfig(
        encoding=asr_pb2.RecognitionConfig.LINEAR_PCM,
        sample_rate_hertz=16000,
        language_code=language,
        model=model,
        enable_word_time_offsets=True,
        enable_speaker_diarization=enable_diarization,
        task="transcribe"
    )
    
    if num_speakers:
        config.diarization_speaker_count = num_speakers
    
    audio = asr_pb2.RecognitionAudio(content=audio_content)
    
    request = asr_pb2.RecognizeRequest(
        config=config,
        audio=audio
    )
    
    # Make request
    logger.info(f"Sending recognition request to {host}:{port}")
    logger.info(f"Config: model={model}, language={language}, diarization={enable_diarization}")
    
    try:
        response = stub.Recognize(request)
        
        # Process response
        logger.info("=" * 80)
        logger.info("TRANSCRIPTION RESULTS")
        logger.info("=" * 80)
        
        for i, result in enumerate(response.results):
            logger.info(f"\nResult {i + 1}:")
            logger.info(f"  Language: {result.language_code}")
            logger.info(f"  Audio duration: {result.audio_processed:.2f}s")
            
            for j, alternative in enumerate(result.alternatives):
                logger.info(f"\n  Alternative {j + 1}:")
                logger.info(f"  Confidence: {alternative.confidence:.4f}")
                logger.info(f"  Transcript: {alternative.transcript}")
                
                if alternative.words:
                    logger.info(f"\n  Word-level details ({len(alternative.words)} words):")
                    logger.info("  " + "-" * 76)
                    logger.info(f"  {'Word':<20} {'Start':<10} {'End':<10} {'Confidence':<12} {'Speaker':<10}")
                    logger.info("  " + "-" * 76)
                    
                    for word in alternative.words[:20]:  # Show first 20 words
                        speaker = f"SPEAKER_{word.speaker_tag:02d}" if word.speaker_tag >= 0 else "N/A"
                        logger.info(
                            f"  {word.word:<20} {word.start_time:<10.3f} "
                            f"{word.end_time:<10.3f} {word.confidence:<12.4f} {speaker:<10}"
                        )
                    
                    if len(alternative.words) > 20:
                        logger.info(f"  ... and {len(alternative.words) - 20} more words")
        
        logger.info("=" * 80)
        return response
        
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
        raise
    finally:
        channel.close()


def streaming_recognize_file(
    audio_file_path: str,
    host: str = "localhost",
    port: int = 50051,
    language: str = "en",
    model: str = "large-v3",
    chunk_size: int = 8192
):
    """
    Transcribe an audio file using streaming gRPC API
    
    Args:
        audio_file_path: Path to audio file
        host: gRPC server host
        port: gRPC server port
        language: Language code
        model: Model name
        chunk_size: Size of audio chunks to stream
    """
    # Create gRPC channel
    channel = grpc.insecure_channel(f"{host}:{port}")
    stub = asr_pb2_grpc.AsrServiceStub(channel)
    
    def request_generator():
        # First request: config
        config = asr_pb2.RecognitionConfig(
            encoding=asr_pb2.RecognitionConfig.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code=language,
            model=model,
            enable_word_time_offsets=True
        )
        
        streaming_config = asr_pb2.StreamingRecognitionConfig(
            config=config,
            interim_results=False
        )
        
        yield asr_pb2.StreamingRecognizeRequest(streaming_config=streaming_config)
        
        # Subsequent requests: audio chunks
        logger.info(f"Reading and streaming audio file: {audio_file_path}")
        with open(audio_file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield asr_pb2.StreamingRecognizeRequest(audio_content=chunk)
    
    logger.info(f"Sending streaming recognition request to {host}:{port}")
    
    try:
        responses = stub.StreamingRecognize(request_generator())
        
        logger.info("=" * 80)
        logger.info("STREAMING TRANSCRIPTION RESULTS")
        logger.info("=" * 80)
        
        for response in responses:
            for result in response.results:
                is_final = result.is_final
                stability = result.stability
                
                logger.info(f"\nResult (final={is_final}, stability={stability:.2f}):")
                logger.info(f"  Language: {result.language_code}")
                
                for alternative in result.alternatives:
                    logger.info(f"  Transcript: {alternative.transcript}")
                    logger.info(f"  Confidence: {alternative.confidence:.4f}")
        
        logger.info("=" * 80)
        
    except grpc.RpcError as e:
        logger.error(f"gRPC error: {e.code()} - {e.details()}")
        raise
    finally:
        channel.close()


def main():
    parser = argparse.ArgumentParser(
        description="gRPC client for WhisperX ASR Service"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file to transcribe"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="gRPC server host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="gRPC server port (default: 50051)"
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Model name (default: large-v3)"
    )
    parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization"
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Number of speakers (if known)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming API"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        logger.error(f"Audio file not found: {args.audio_file}")
        sys.exit(1)
    
    try:
        if args.streaming:
            streaming_recognize_file(
                audio_file_path=args.audio_file,
                host=args.host,
                port=args.port,
                language=args.language,
                model=args.model
            )
        else:
            recognize_file(
                audio_file_path=args.audio_file,
                host=args.host,
                port=args.port,
                language=args.language,
                model=args.model,
                enable_diarization=not args.no_diarization,
                num_speakers=args.num_speakers
            )
    except Exception as e:
        logger.error(f"Failed to transcribe: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
