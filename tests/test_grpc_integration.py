#!/usr/bin/env python3
"""
Basic integration tests for gRPC server
These tests verify the structure and imports work correctly
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestGRPCStructure(unittest.TestCase):
    """Test that gRPC components are properly structured"""
    
    def test_import_proto_definitions(self):
        """Test that proto definitions can be imported"""
        try:
            from app.grpc_generated import asr_pb2, asr_pb2_grpc
            self.assertIsNotNone(asr_pb2)
            self.assertIsNotNone(asr_pb2_grpc)
        except ImportError as e:
            self.fail(f"Failed to import proto definitions: {e}")
    
    def test_import_asr_interface(self):
        """Test that ASR interface can be imported"""
        try:
            from app.asr_interface import (
                ASRInterface, 
                ASRConfig, 
                ASRResult, 
                Segment, 
                WordInfo
            )
            self.assertIsNotNone(ASRInterface)
            self.assertIsNotNone(ASRConfig)
            self.assertIsNotNone(ASRResult)
        except ImportError as e:
            self.fail(f"Failed to import ASR interface: {e}")
    
    def test_asr_config_defaults(self):
        """Test ASR config has correct defaults"""
        from app.asr_interface import ASRConfig
        
        config = ASRConfig()
        self.assertEqual(config.task, "transcribe")
        self.assertEqual(config.model, "large-v3")
        self.assertTrue(config.enable_diarization)
        self.assertTrue(config.enable_word_timestamps)
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.diarization_model, "pyannote/speaker-diarization-community-1")
    
    def test_recognition_config_enum(self):
        """Test RecognitionConfig enum values exist"""
        from app.grpc_generated import asr_pb2
        
        # Test encoding enum
        self.assertTrue(hasattr(asr_pb2.RecognitionConfig, 'LINEAR_PCM'))
        self.assertTrue(hasattr(asr_pb2.RecognitionConfig, 'FLAC'))
        self.assertTrue(hasattr(asr_pb2.RecognitionConfig, 'MP3'))
    
    def test_request_message_structure(self):
        """Test that request messages have expected structure"""
        from app.grpc_generated import asr_pb2
        
        # Create a RecognizeRequest
        config = asr_pb2.RecognitionConfig(
            encoding=asr_pb2.RecognitionConfig.LINEAR_PCM,
            sample_rate_hertz=16000,
            language_code="en"
        )
        audio = asr_pb2.RecognitionAudio(content=b"dummy_audio")
        request = asr_pb2.RecognizeRequest(config=config, audio=audio)
        
        self.assertEqual(request.config.language_code, "en")
        self.assertEqual(request.audio.content, b"dummy_audio")
    
    def test_grpc_servicer_creation(self):
        """Test that gRPC servicer can be created"""
        try:
            # Mock dependencies before importing
            sys.modules['whisperx'] = Mock()
            sys.modules['whisperx.diarize'] = Mock()
            sys.modules['torch'] = Mock()
            sys.modules['numpy'] = Mock()
            
            from app.grpc_server import AsrServiceServicer
            from app.asr_interface import ASRInterface
            
            # Create a mock ASR engine
            mock_engine = Mock(spec=ASRInterface)
            
            # Create servicer
            servicer = AsrServiceServicer(mock_engine)
            self.assertIsNotNone(servicer)
            self.assertEqual(servicer.asr_engine, mock_engine)
        except Exception as e:
            self.fail(f"Failed to create gRPC servicer: {e}")
        finally:
            # Clean up mocks
            for mod in ['whisperx', 'whisperx.diarize', 'torch', 'numpy']:
                if mod in sys.modules:
                    del sys.modules[mod]
    
    def test_encoding_string_conversion(self):
        """Test encoding enum to string conversion"""
        try:
            # Mock dependencies
            sys.modules['whisperx'] = Mock()
            sys.modules['whisperx.diarize'] = Mock()
            sys.modules['torch'] = Mock()
            sys.modules['numpy'] = Mock()
            
            from app.grpc_generated import asr_pb2
            from app.grpc_server import AsrServiceServicer
            from app.asr_interface import ASRInterface
            
            mock_engine = Mock(spec=ASRInterface)
            servicer = AsrServiceServicer(mock_engine)
            
            # Test conversion
            linear_pcm = servicer._encoding_to_string(asr_pb2.RecognitionConfig.LINEAR_PCM)
            self.assertEqual(linear_pcm, "LINEAR_PCM")
            
            mp3 = servicer._encoding_to_string(asr_pb2.RecognitionConfig.MP3)
            self.assertEqual(mp3, "MP3")
        finally:
            # Clean up mocks
            for mod in ['whisperx', 'whisperx.diarize', 'torch', 'numpy']:
                if mod in sys.modules:
                    del sys.modules[mod]


class TestModelAbstraction(unittest.TestCase):
    """Test the model abstraction layer"""
    
    def test_asr_interface_is_abstract(self):
        """Test that ASRInterface cannot be instantiated directly"""
        from app.asr_interface import ASRInterface
        
        with self.assertRaises(TypeError):
            ASRInterface()
    
    def test_asr_config_creation(self):
        """Test creating ASR config with custom values"""
        from app.asr_interface import ASRConfig
        
        config = ASRConfig(
            language="es",
            model="medium",
            enable_diarization=False,
            num_speakers=3
        )
        
        self.assertEqual(config.language, "es")
        self.assertEqual(config.model, "medium")
        self.assertFalse(config.enable_diarization)
        self.assertEqual(config.num_speakers, 3)
    
    def test_word_info_structure(self):
        """Test WordInfo dataclass structure"""
        from app.asr_interface import WordInfo
        
        word = WordInfo(
            word="hello",
            start=1.5,
            end=2.0,
            confidence=0.95,
            speaker="SPEAKER_00"
        )
        
        self.assertEqual(word.word, "hello")
        self.assertEqual(word.start, 1.5)
        self.assertEqual(word.end, 2.0)
        self.assertEqual(word.confidence, 0.95)
        self.assertEqual(word.speaker, "SPEAKER_00")
    
    def test_segment_structure(self):
        """Test Segment dataclass structure"""
        from app.asr_interface import Segment, WordInfo
        
        words = [
            WordInfo(word="hello", start=0.0, end=0.5, confidence=0.9),
            WordInfo(word="world", start=0.5, end=1.0, confidence=0.95)
        ]
        
        segment = Segment(
            text="hello world",
            start=0.0,
            end=1.0,
            words=words,
            speaker="SPEAKER_00"
        )
        
        self.assertEqual(segment.text, "hello world")
        self.assertEqual(len(segment.words), 2)
        self.assertEqual(segment.speaker, "SPEAKER_00")


if __name__ == "__main__":
    # Run tests
    unittest.main()
