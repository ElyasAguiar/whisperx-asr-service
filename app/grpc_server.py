"""
gRPC Server for ASR Service
Compatible with NVIDIA Riva ASR protocol
"""

import os
import io
import logging
import wave
from typing import Iterator

import grpc
from concurrent import futures

from .grpc_generated import asr_pb2, asr_pb2_grpc
from .asr_interface import ASRInterface, ASRConfig
from .whisperx_asr import WhisperXASR

logger = logging.getLogger(__name__)


class AsrServiceServicer(asr_pb2_grpc.AsrServiceServicer):
    """gRPC servicer implementation for ASR"""
    
    def __init__(self, asr_engine: ASRInterface):
        self.asr_engine = asr_engine
        logger.info("gRPC ASR Servicer initialized")
    
    def _config_from_proto(self, config: asr_pb2.RecognitionConfig) -> ASRConfig:
        """Convert protobuf config to ASRConfig"""
        return ASRConfig(
            language=config.language_code if config.language_code else None,
            task=config.task if config.task else "transcribe",
            model=config.model if config.model else "large-v3",
            enable_diarization=config.enable_speaker_diarization,
            enable_word_timestamps=config.enable_word_time_offsets,
            num_speakers=config.diarization_speaker_count if config.diarization_speaker_count > 0 else None,
            min_speakers=config.min_speaker_count if config.min_speaker_count > 0 else None,
            max_speakers=config.max_speaker_count if config.max_speaker_count > 0 else None,
            initial_prompt=config.initial_prompt if config.initial_prompt else None,
            sample_rate=config.sample_rate_hertz if config.sample_rate_hertz > 0 else 16000,
            audio_encoding=self._encoding_to_string(config.encoding)
        )
    
    def _encoding_to_string(self, encoding) -> str:
        """Convert protobuf encoding enum to string"""
        encoding_map = {
            asr_pb2.RecognitionConfig.LINEAR_PCM: "LINEAR_PCM",
            asr_pb2.RecognitionConfig.FLAC: "FLAC",
            asr_pb2.RecognitionConfig.MP3: "MP3",
            asr_pb2.RecognitionConfig.OGG_OPUS: "OGG_OPUS",
        }
        return encoding_map.get(encoding, "LINEAR_PCM")
    
    def _convert_audio_to_wav(self, audio_bytes: bytes, sample_rate: int, encoding: str) -> bytes:
        """Convert audio bytes to WAV format for processing"""
        # If already in a format that whisperx can handle, return as-is
        # For LINEAR_PCM, wrap in WAV container
        if encoding == "LINEAR_PCM":
            # Create WAV file in memory
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)
            return wav_io.getvalue()
        else:
            # For other formats, assume they're already in a valid format
            return audio_bytes
    
    def Recognize(self, request: asr_pb2.RecognizeRequest, context) -> asr_pb2.RecognizeResponse:
        """
        Perform speech recognition on audio file
        
        Args:
            request: Recognition request with audio and config
            context: gRPC context
            
        Returns:
            Recognition response with transcription
        """
        try:
            logger.info("Received Recognize request")
            
            # Extract configuration
            config = self._config_from_proto(request.config)
            
            # Extract audio
            if request.audio.content:
                audio_bytes = request.audio.content
            elif request.audio.uri:
                # URI-based audio not implemented yet
                context.abort(
                    grpc.StatusCode.UNIMPLEMENTED,
                    "URI-based audio input not yet supported"
                )
                return
            else:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No audio content provided")
                return
            
            # Convert audio to WAV format if needed
            audio_bytes = self._convert_audio_to_wav(audio_bytes, config.sample_rate, config.audio_encoding)
            
            # Process audio
            logger.info(f"Processing audio: {len(audio_bytes)} bytes, model: {config.model}, language: {config.language}")
            result = self.asr_engine.transcribe(audio_bytes, config)
            logger.info(f"Transcription complete: {len(result.segments)} segments")
            
            # Build response
            response = asr_pb2.RecognizeResponse()
            
            # Create a single result with all segments
            speech_result = response.results.add()
            speech_result.language_code = result.language
            speech_result.audio_processed = result.duration
            
            # Create alternative with full transcript
            alternative = speech_result.alternatives.add()
            alternative.transcript = " ".join([seg.text for seg in result.segments])
            alternative.confidence = 1.0
            
            # Add word-level information
            for segment in result.segments:
                for word_info in segment.words:
                    word = alternative.words.add()
                    word.word = word_info.word
                    word.start_time = word_info.start
                    word.end_time = word_info.end
                    word.confidence = word_info.confidence
                    
                    # Add speaker tag if available
                    if word_info.speaker:
                        # Extract speaker number from speaker tag (e.g., "SPEAKER_00" -> 0)
                        try:
                            speaker_num = int(word_info.speaker.split("_")[-1])
                            word.speaker_tag = speaker_num
                        except (ValueError, IndexError):
                            word.speaker_tag = 0
            
            logger.info("Recognition response prepared successfully")
            return response
            
        except Exception as e:
            logger.error(f"Recognition failed: {str(e)}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Recognition failed: {str(e)}")
    
    def StreamingRecognize(
        self,
        request_iterator: Iterator[asr_pb2.StreamingRecognizeRequest],
        context
    ) -> Iterator[asr_pb2.StreamingRecognizeResponse]:
        """
        Perform streaming speech recognition
        
        Args:
            request_iterator: Iterator of streaming requests
            context: gRPC context
            
        Yields:
            Streaming recognition responses
        """
        try:
            logger.info("Received StreamingRecognize request")
            
            config = None
            audio_chunks = []
            
            # Collect audio chunks
            for request in request_iterator:
                if request.HasField("streaming_config"):
                    # First message contains config
                    config = self._config_from_proto(request.streaming_config.config)
                    logger.info(f"Streaming config received: model={config.model}, language={config.language}")
                elif request.audio_content:
                    # Subsequent messages contain audio
                    audio_chunks.append(request.audio_content)
            
            if not config:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No streaming config provided")
                return
            
            if not audio_chunks:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "No audio content received")
                return
            
            # Combine all audio chunks
            audio_bytes = b"".join(audio_chunks)
            logger.info(f"Received {len(audio_chunks)} audio chunks, total {len(audio_bytes)} bytes")
            
            # Convert audio to WAV format if needed
            audio_bytes = self._convert_audio_to_wav(audio_bytes, config.sample_rate, config.audio_encoding)
            
            # Process audio
            result = self.asr_engine.transcribe(audio_bytes, config)
            logger.info(f"Streaming transcription complete: {len(result.segments)} segments")
            
            # Build streaming response
            # For simplicity, we return one final response with all segments
            # A more sophisticated implementation could return interim results
            response = asr_pb2.StreamingRecognizeResponse()
            
            stream_result = response.results.add()
            stream_result.is_final = True
            stream_result.stability = 1.0
            stream_result.language_code = result.language
            
            alternative = stream_result.alternatives.add()
            alternative.transcript = " ".join([seg.text for seg in result.segments])
            alternative.confidence = 1.0
            
            # Add word-level information
            for segment in result.segments:
                for word_info in segment.words:
                    word = alternative.words.add()
                    word.word = word_info.word
                    word.start_time = word_info.start
                    word.end_time = word_info.end
                    word.confidence = word_info.confidence
                    
                    if word_info.speaker:
                        try:
                            speaker_num = int(word_info.speaker.split("_")[-1])
                            word.speaker_tag = speaker_num
                        except (ValueError, IndexError):
                            word.speaker_tag = 0
            
            yield response
            logger.info("Streaming recognition response sent")
            
        except Exception as e:
            logger.error(f"Streaming recognition failed: {str(e)}", exc_info=True)
            context.abort(grpc.StatusCode.INTERNAL, f"Streaming recognition failed: {str(e)}")


def serve_grpc(
    port: int = 50051,
    asr_engine: ASRInterface = None,
    max_workers: int = 10
):
    """
    Start the gRPC server
    
    Args:
        port: Port to listen on
        asr_engine: ASR engine to use (defaults to WhisperXASR)
        max_workers: Maximum number of worker threads
    """
    # Create ASR engine if not provided
    if asr_engine is None:
        device = os.getenv("DEVICE", "cuda")
        compute_type = os.getenv("COMPUTE_TYPE", "float16")
        batch_size = int(os.getenv("BATCH_SIZE", "16"))
        hf_token = os.getenv("HF_TOKEN", None)
        cache_dir = os.getenv("CACHE_DIR", "/.cache")
        
        asr_engine = WhisperXASR(
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            hf_token=hf_token,
            cache_dir=cache_dir
        )
    
    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    asr_pb2_grpc.add_AsrServiceServicer_to_server(
        AsrServiceServicer(asr_engine),
        server
    )
    
    # Bind to port
    server.add_insecure_port(f"[::]:{port}")
    
    # Start server
    server.start()
    logger.info(f"gRPC server started on port {port}")
    
    return server


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    port = int(os.getenv("GRPC_PORT", "50051"))
    server = serve_grpc(port=port)
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server...")
        server.stop(0)
