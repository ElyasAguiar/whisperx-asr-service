from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class RecognitionConfig(_message.Message):
    __slots__ = (
        "encoding",
        "sample_rate_hertz",
        "audio_channel_count",
        "language_code",
        "max_alternatives",
        "model",
        "enable_automatic_punctuation",
        "enable_speaker_diarization",
        "diarization_speaker_count",
        "min_speaker_count",
        "max_speaker_count",
        "enable_word_time_offsets",
        "task",
        "initial_prompt",
    )

    class AudioEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        ENCODING_UNSPECIFIED: _ClassVar[RecognitionConfig.AudioEncoding]
        LINEAR_PCM: _ClassVar[RecognitionConfig.AudioEncoding]
        FLAC: _ClassVar[RecognitionConfig.AudioEncoding]
        MULAW: _ClassVar[RecognitionConfig.AudioEncoding]
        ALAW: _ClassVar[RecognitionConfig.AudioEncoding]
        OGG_OPUS: _ClassVar[RecognitionConfig.AudioEncoding]
        MP3: _ClassVar[RecognitionConfig.AudioEncoding]

    ENCODING_UNSPECIFIED: RecognitionConfig.AudioEncoding
    LINEAR_PCM: RecognitionConfig.AudioEncoding
    FLAC: RecognitionConfig.AudioEncoding
    MULAW: RecognitionConfig.AudioEncoding
    ALAW: RecognitionConfig.AudioEncoding
    OGG_OPUS: RecognitionConfig.AudioEncoding
    MP3: RecognitionConfig.AudioEncoding
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_HERTZ_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CHANNEL_COUNT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CODE_FIELD_NUMBER: _ClassVar[int]
    MAX_ALTERNATIVES_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    ENABLE_AUTOMATIC_PUNCTUATION_FIELD_NUMBER: _ClassVar[int]
    ENABLE_SPEAKER_DIARIZATION_FIELD_NUMBER: _ClassVar[int]
    DIARIZATION_SPEAKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    MIN_SPEAKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEAKER_COUNT_FIELD_NUMBER: _ClassVar[int]
    ENABLE_WORD_TIME_OFFSETS_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    INITIAL_PROMPT_FIELD_NUMBER: _ClassVar[int]
    encoding: RecognitionConfig.AudioEncoding
    sample_rate_hertz: int
    audio_channel_count: int
    language_code: str
    max_alternatives: int
    model: str
    enable_automatic_punctuation: bool
    enable_speaker_diarization: bool
    diarization_speaker_count: int
    min_speaker_count: int
    max_speaker_count: int
    enable_word_time_offsets: bool
    task: str
    initial_prompt: str
    def __init__(
        self,
        encoding: _Optional[_Union[RecognitionConfig.AudioEncoding, str]] = ...,
        sample_rate_hertz: _Optional[int] = ...,
        audio_channel_count: _Optional[int] = ...,
        language_code: _Optional[str] = ...,
        max_alternatives: _Optional[int] = ...,
        model: _Optional[str] = ...,
        enable_automatic_punctuation: bool = ...,
        enable_speaker_diarization: bool = ...,
        diarization_speaker_count: _Optional[int] = ...,
        min_speaker_count: _Optional[int] = ...,
        max_speaker_count: _Optional[int] = ...,
        enable_word_time_offsets: bool = ...,
        task: _Optional[str] = ...,
        initial_prompt: _Optional[str] = ...,
    ) -> None: ...

class RecognitionAudio(_message.Message):
    __slots__ = ("content", "uri")
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    content: bytes
    uri: str
    def __init__(
        self, content: _Optional[bytes] = ..., uri: _Optional[str] = ...
    ) -> None: ...

class RecognizeRequest(_message.Message):
    __slots__ = ("config", "audio")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    config: RecognitionConfig
    audio: RecognitionAudio
    def __init__(
        self,
        config: _Optional[_Union[RecognitionConfig, _Mapping]] = ...,
        audio: _Optional[_Union[RecognitionAudio, _Mapping]] = ...,
    ) -> None: ...

class StreamingRecognizeRequest(_message.Message):
    __slots__ = ("streaming_config", "audio_content")
    STREAMING_CONFIG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CONTENT_FIELD_NUMBER: _ClassVar[int]
    streaming_config: StreamingRecognitionConfig
    audio_content: bytes
    def __init__(
        self,
        streaming_config: _Optional[_Union[StreamingRecognitionConfig, _Mapping]] = ...,
        audio_content: _Optional[bytes] = ...,
    ) -> None: ...

class StreamingRecognitionConfig(_message.Message):
    __slots__ = ("config", "interim_results")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    INTERIM_RESULTS_FIELD_NUMBER: _ClassVar[int]
    config: RecognitionConfig
    interim_results: bool
    def __init__(
        self,
        config: _Optional[_Union[RecognitionConfig, _Mapping]] = ...,
        interim_results: bool = ...,
    ) -> None: ...

class WordInfo(_message.Message):
    __slots__ = ("start_time", "end_time", "word", "confidence", "speaker_tag")
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    WORD_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SPEAKER_TAG_FIELD_NUMBER: _ClassVar[int]
    start_time: float
    end_time: float
    word: str
    confidence: float
    speaker_tag: int
    def __init__(
        self,
        start_time: _Optional[float] = ...,
        end_time: _Optional[float] = ...,
        word: _Optional[str] = ...,
        confidence: _Optional[float] = ...,
        speaker_tag: _Optional[int] = ...,
    ) -> None: ...

class SpeechRecognitionAlternative(_message.Message):
    __slots__ = ("transcript", "confidence", "words")
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    transcript: str
    confidence: float
    words: _containers.RepeatedCompositeFieldContainer[WordInfo]
    def __init__(
        self,
        transcript: _Optional[str] = ...,
        confidence: _Optional[float] = ...,
        words: _Optional[_Iterable[_Union[WordInfo, _Mapping]]] = ...,
    ) -> None: ...

class SpeechRecognitionResult(_message.Message):
    __slots__ = ("alternatives", "channel_tag", "language_code", "audio_processed")
    ALTERNATIVES_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_TAG_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CODE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PROCESSED_FIELD_NUMBER: _ClassVar[int]
    alternatives: _containers.RepeatedCompositeFieldContainer[
        SpeechRecognitionAlternative
    ]
    channel_tag: int
    language_code: str
    audio_processed: float
    def __init__(
        self,
        alternatives: _Optional[
            _Iterable[_Union[SpeechRecognitionAlternative, _Mapping]]
        ] = ...,
        channel_tag: _Optional[int] = ...,
        language_code: _Optional[str] = ...,
        audio_processed: _Optional[float] = ...,
    ) -> None: ...

class RecognizeResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[SpeechRecognitionResult]
    def __init__(
        self,
        results: _Optional[_Iterable[_Union[SpeechRecognitionResult, _Mapping]]] = ...,
    ) -> None: ...

class StreamingRecognizeResponse(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[StreamingRecognitionResult]
    def __init__(
        self,
        results: _Optional[
            _Iterable[_Union[StreamingRecognitionResult, _Mapping]]
        ] = ...,
    ) -> None: ...

class StreamingRecognitionResult(_message.Message):
    __slots__ = (
        "alternatives",
        "is_final",
        "stability",
        "channel_tag",
        "language_code",
    )
    ALTERNATIVES_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    STABILITY_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_TAG_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_CODE_FIELD_NUMBER: _ClassVar[int]
    alternatives: _containers.RepeatedCompositeFieldContainer[
        SpeechRecognitionAlternative
    ]
    is_final: bool
    stability: float
    channel_tag: int
    language_code: str
    def __init__(
        self,
        alternatives: _Optional[
            _Iterable[_Union[SpeechRecognitionAlternative, _Mapping]]
        ] = ...,
        is_final: bool = ...,
        stability: _Optional[float] = ...,
        channel_tag: _Optional[int] = ...,
        language_code: _Optional[str] = ...,
    ) -> None: ...
