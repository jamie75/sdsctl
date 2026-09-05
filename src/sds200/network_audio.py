from __future__ import annotations

import logging
import socket
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from time import monotonic
from typing import Protocol

from .audio import AudioChunk, AudioChunkHandler
from .events import EventBus
from .exceptions import ScannerConnectionError
from .pcmu import PcmuPacket, PcmuPacketHandler
from .rtp import (
    RtpPacket,
    RtpProtocolError,
    RtpSequenceTracker,
    RtpTimestampTracker,
)
from .rtsp import (
    DEFAULT_AUDIO_PATH,
    DEFAULT_RTSP_PORT,
    RtpTransportInfo,
    RtspClient,
    RtspProtocolError,
)
from .socket_utils import (
    LocalAddressResolver,
    normalize_local_ipv4_bind_address,
    resolve_local_ipv4_address,
)

logger = logging.getLogger(__name__)
MAX_RTP_DATAGRAM_SIZE = 65535
DEFAULT_RTP_INACTIVITY_TIMEOUT = 60.0
DEFAULT_AUDIO_RECOVERY_ATTEMPTS = 3
DEFAULT_AUDIO_RECOVERY_BACKOFF = 1.0


class AudioDatagramSocketLike(Protocol):
    def settimeout(self, value: float | None) -> None: ...

    def bind(self, address: tuple[str, int]) -> None: ...

    def getsockname(self) -> tuple[str, int]: ...

    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]: ...

    def close(self) -> None: ...


class RtspSessionClientLike(Protocol):
    def start(self, client_port: int) -> RtpTransportInfo: ...

    def get_parameter(self) -> object: ...

    def teardown(self) -> object: ...

    def close(self) -> None: ...


AudioDatagramSocketFactory = Callable[[int, int], AudioDatagramSocketLike]
RtspSessionClientFactory = Callable[[str, int, str, float], RtspSessionClientLike]


@dataclass(frozen=True, slots=True)
class NetworkAudioStatistics:
    """Snapshot of one network-audio transport session."""

    sessions_started: int = 0
    datagrams_received: int = 0
    bytes_received: int = 0
    packets_delivered: int = 0
    payload_bytes_delivered: int = 0
    sequence_gaps: int = 0
    packets_lost: int = 0
    duplicate_packets: int = 0
    late_packets: int = 0
    malformed_packets: int = 0
    unexpected_source_packets: int = 0
    ssrc_mismatch_packets: int = 0
    timestamp_discontinuities: int = 0
    timestamp_samples_missing: int = 0
    timestamp_backwards: int = 0
    receive_errors: int = 0
    callback_errors: int = 0
    keepalives_sent: int = 0
    keepalive_failures: int = 0
    teardowns_sent: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    last_timestamp: int | None = None
    ssrc: int | None = None


@dataclass(slots=True)
class _MutableNetworkAudioStatistics:
    sessions_started: int = 0
    datagrams_received: int = 0
    bytes_received: int = 0
    packets_delivered: int = 0
    payload_bytes_delivered: int = 0
    sequence_gaps: int = 0
    packets_lost: int = 0
    duplicate_packets: int = 0
    late_packets: int = 0
    malformed_packets: int = 0
    unexpected_source_packets: int = 0
    ssrc_mismatch_packets: int = 0
    timestamp_discontinuities: int = 0
    timestamp_samples_missing: int = 0
    timestamp_backwards: int = 0
    receive_errors: int = 0
    callback_errors: int = 0
    keepalives_sent: int = 0
    keepalive_failures: int = 0
    teardowns_sent: int = 0
    first_sequence: int | None = None
    last_sequence: int | None = None
    last_timestamp: int | None = None
    ssrc: int | None = None

    def snapshot(self) -> NetworkAudioStatistics:
        return NetworkAudioStatistics(
            sessions_started=self.sessions_started,
            datagrams_received=self.datagrams_received,
            bytes_received=self.bytes_received,
            packets_delivered=self.packets_delivered,
            payload_bytes_delivered=self.payload_bytes_delivered,
            sequence_gaps=self.sequence_gaps,
            packets_lost=self.packets_lost,
            duplicate_packets=self.duplicate_packets,
            late_packets=self.late_packets,
            malformed_packets=self.malformed_packets,
            unexpected_source_packets=self.unexpected_source_packets,
            ssrc_mismatch_packets=self.ssrc_mismatch_packets,
            timestamp_discontinuities=self.timestamp_discontinuities,
            timestamp_samples_missing=self.timestamp_samples_missing,
            timestamp_backwards=self.timestamp_backwards,
            receive_errors=self.receive_errors,
            callback_errors=self.callback_errors,
            keepalives_sent=self.keepalives_sent,
            keepalive_failures=self.keepalive_failures,
            teardowns_sent=self.teardowns_sent,
            first_sequence=self.first_sequence,
            last_sequence=self.last_sequence,
            last_timestamp=self.last_timestamp,
            ssrc=self.ssrc,
        )


def default_audio_datagram_socket_factory(
    family: int,
    socket_type: int,
) -> AudioDatagramSocketLike:
    return socket.socket(family, socket_type)


def default_rtsp_session_client_factory(
    host: str,
    port: int,
    path: str,
    timeout: float,
) -> RtspSessionClientLike:
    return RtspClient(host, port=port, path=path, timeout=timeout)


class NetworkAudioTransport:
    """SDS200 RTSP/RTP network audio transport emitting raw PCMU payloads."""

    def __init__(
        self,
        host: str,
        *,
        rtsp_port: int = DEFAULT_RTSP_PORT,
        path: str = DEFAULT_AUDIO_PATH,
        local_host: str | None = None,
        local_port: int = 0,
        read_timeout: float = 0.2,
        rtsp_timeout: float = 5.0,
        keepalive_interval: float = 15.0,
        inactivity_timeout: float = DEFAULT_RTP_INACTIVITY_TIMEOUT,
        recovery_attempts: int = DEFAULT_AUDIO_RECOVERY_ATTEMPTS,
        recovery_backoff: float = DEFAULT_AUDIO_RECOVERY_BACKOFF,
        datagram_socket_factory: AudioDatagramSocketFactory = (
            default_audio_datagram_socket_factory
        ),
        rtsp_client_factory: RtspSessionClientFactory = (
            default_rtsp_session_client_factory
        ),
        local_address_resolver: LocalAddressResolver = resolve_local_ipv4_address,
    ) -> None:
        if not host.strip():
            raise ValueError("Audio host must not be empty.")
        if not 1 <= rtsp_port <= 65535:
            raise ValueError("RTSP port must be between 1 and 65535.")
        if not 0 <= local_port <= 65535:
            raise ValueError("Local RTP port must be between 0 and 65535.")
        normalized_local_host = normalize_local_ipv4_bind_address(
            local_host,
            description="Local RTP address",
        )
        if read_timeout <= 0:
            raise ValueError("Audio read timeout must be greater than zero.")
        if rtsp_timeout <= 0:
            raise ValueError("RTSP timeout must be greater than zero.")
        if keepalive_interval <= 0:
            raise ValueError("RTSP keepalive interval must be greater than zero.")
        if inactivity_timeout <= 0:
            raise ValueError("RTP inactivity timeout must be greater than zero.")
        if type(recovery_attempts) is not int or recovery_attempts <= 0:
            raise ValueError("Audio recovery attempts must be a positive integer.")
        if recovery_backoff <= 0:
            raise ValueError("Audio recovery backoff must be greater than zero.")

        self.host = host
        self.rtsp_port = rtsp_port
        self.path = path
        self.local_host = normalized_local_host
        self.local_port = local_port
        self.read_timeout = read_timeout
        self.rtsp_timeout = rtsp_timeout
        self.keepalive_interval = keepalive_interval
        self.inactivity_timeout = inactivity_timeout
        self.recovery_attempts = recovery_attempts
        self.recovery_backoff = recovery_backoff
        self._datagram_socket_factory = datagram_socket_factory
        self._rtsp_client_factory = rtsp_client_factory
        self._local_address_resolver = local_address_resolver
        self.events = EventBus()
        self._rtp_socket: AudioDatagramSocketLike | None = None
        self._rtsp_client: RtspSessionClientLike | None = None
        self._handler: AudioChunkHandler | None = None
        self._receiver_thread: threading.Thread | None = None
        self._keepalive_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lifecycle_lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._rtsp_lock = threading.Lock()
        self._statistics_lock = threading.RLock()
        self._statistics = _MutableNetworkAudioStatistics()
        self._sequence_tracker = RtpSequenceTracker()
        self._timestamp_tracker = RtpTimestampTracker()
        self._expected_source: tuple[str, int] | None = None
        self._expected_ssrc: int | None = None
        self._last_rtp_packet_monotonic: float | None = None
        self._rtp_watchdog_started_monotonic: float | None = None
        self._rtp_packet_seen_this_session = False
        self._rtp_ever_received = False
        self._last_keepalive_monotonic: float | None = None
        self._audio_recovery_state = "stopped"
        self._audio_recovery_count = 0
        self._last_audio_recovery_error: str | None = None
        self._recovery_active = False
        self._recovery_thread: threading.Thread | None = None

    @property
    def endpoint(self) -> str:
        authority = self.host if self.rtsp_port == DEFAULT_RTSP_PORT else (
            f"{self.host}:{self.rtsp_port}"
        )
        return f"rtsp://{authority}{self.path}"

    @property
    def running(self) -> bool:
        with self._state_lock:
            receiver = self._receiver_thread
            keepalive = self._keepalive_thread
            return (
                self._rtp_socket is not None
                and self._rtsp_client is not None
                and receiver is not None
                and receiver.is_alive()
                and keepalive is not None
                and keepalive.is_alive()
                and not self._stop.is_set()
            )

    @property
    def rtp_receiver_alive(self) -> bool:
        with self._state_lock:
            return (
                self._receiver_thread is not None
                and self._receiver_thread.is_alive()
            )

    @property
    def rtsp_keepalive_alive(self) -> bool:
        with self._state_lock:
            return (
                self._keepalive_thread is not None
                and self._keepalive_thread.is_alive()
            )

    @property
    def last_rtp_packet_age_seconds(self) -> float | None:
        with self._state_lock:
            last_packet = self._last_rtp_packet_monotonic
        if last_packet is None:
            return None
        return max(0.0, monotonic() - last_packet)

    @property
    def rtp_active(self) -> bool:
        age = self.last_rtp_packet_age_seconds
        return (
            age is not None
            and age < self.inactivity_timeout
            and self._rtp_packet_seen_this_session
            and self.rtp_receiver_alive
            and self.audio_recovery_state == "healthy"
        )

    @property
    def audio_recovery_state(self) -> str:
        with self._state_lock:
            return self._audio_recovery_state

    @property
    def audio_recovery_count(self) -> int:
        with self._state_lock:
            return self._audio_recovery_count

    @property
    def last_audio_recovery_error(self) -> str | None:
        with self._state_lock:
            return self._last_audio_recovery_error

    @property
    def statistics(self) -> NetworkAudioStatistics:
        with self._statistics_lock:
            return self._statistics.snapshot()

    def on_packet(
        self,
        callback: PcmuPacketHandler,
    ) -> Callable[[], None]:
        """Subscribe to accepted RTP PCMU packets before PCM decoding."""

        return self.events.subscribe("packet", callback)

    def start(self, handler: AudioChunkHandler) -> None:
        with self._lifecycle_lock:
            self._start_session(handler, reset_statistics=True)

    def _start_session(
        self,
        handler: AudioChunkHandler,
        *,
        reset_statistics: bool,
    ) -> None:
        with self._state_lock:
            if self._rtp_socket is not None:
                return
            self._handler = handler
            self._stop.clear()
            self._sequence_tracker.reset()
            self._timestamp_tracker.reset()
            if reset_statistics:
                self._rtp_ever_received = False
            self._last_rtp_packet_monotonic = None
            self._rtp_watchdog_started_monotonic = (
                monotonic() if not reset_statistics and self._rtp_ever_received else None
            )
            self._rtp_packet_seen_this_session = False
            self._last_keepalive_monotonic = None
            self._audio_recovery_state = "starting"
            if reset_statistics:
                self._last_audio_recovery_error = None
        if reset_statistics:
            with self._statistics_lock:
                self._statistics = _MutableNetworkAudioStatistics()

        rtp_socket: AudioDatagramSocketLike | None = None
        rtsp_client: RtspSessionClientLike | None = None
        negotiated: RtpTransportInfo | None = None
        try:
            rtp_socket = self._datagram_socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
            rtp_socket.settimeout(self.read_timeout)
            bind_host = self.local_host
            if bind_host is None:
                bind_host = self._local_address_resolver(
                    self.host,
                    self.rtsp_port,
                )
            rtp_socket.bind((bind_host, self.local_port))
            client_port = rtp_socket.getsockname()[1]
            if not 1 <= client_port <= 65535:
                raise ScannerConnectionError(
                    f"Could not allocate an RTP client port for {self.endpoint}."
                )

            rtsp_client = self._rtsp_client_factory(
                self.host,
                self.rtsp_port,
                self.path,
                self.rtsp_timeout,
            )
            negotiated = rtsp_client.start(client_port)
        except (OSError, RtspProtocolError, ScannerConnectionError) as exc:
            if rtsp_client is not None:
                with suppress(Exception):
                    rtsp_client.close()
            if rtp_socket is not None:
                with suppress(OSError):
                    rtp_socket.close()
            with self._state_lock:
                self._handler = None
                self._audio_recovery_state = "failed"
                self._last_audio_recovery_error = exc.__class__.__name__
            raise ScannerConnectionError(
                f"Could not start SDS200 network audio at {self.endpoint}."
            ) from exc

        assert rtp_socket is not None
        assert rtsp_client is not None
        assert negotiated is not None
        self._expected_source = (negotiated.source, negotiated.server_port)
        self._expected_ssrc = negotiated.ssrc
        with self._statistics_lock:
            self._statistics.sessions_started += 1
            self._statistics.ssrc = negotiated.ssrc
        with self._state_lock:
            self._rtp_socket = rtp_socket
            self._rtsp_client = rtsp_client
            self._receiver_thread = threading.Thread(
                target=self._receiver_loop,
                name="sds200-audio-rtp-reader",
                daemon=True,
            )
            self._keepalive_thread = threading.Thread(
                target=self._keepalive_loop,
                name="sds200-audio-rtsp-keepalive",
                daemon=True,
            )
            receiver = self._receiver_thread
            keepalive = self._keepalive_thread
            self._audio_recovery_state = "healthy"
        assert receiver is not None
        assert keepalive is not None
        receiver.start()
        keepalive.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._stop.set()
            with self._state_lock:
                rtp_socket, self._rtp_socket = self._rtp_socket, None
                rtsp_client, self._rtsp_client = self._rtsp_client, None
                receiver, self._receiver_thread = self._receiver_thread, None
                keepalive, self._keepalive_thread = self._keepalive_thread, None

            if rtsp_client is not None:
                with self._rtsp_lock:
                    with suppress(Exception):
                        rtsp_client.teardown()
                        with self._statistics_lock:
                            self._statistics.teardowns_sent += 1
                    with suppress(Exception):
                        rtsp_client.close()
            if rtp_socket is not None:
                with suppress(OSError):
                    rtp_socket.close()

            current = threading.current_thread()
            for thread in (receiver, keepalive):
                if thread is not None and thread is not current:
                    thread.join(timeout=max(1.0, self.read_timeout * 4))
            with self._state_lock:
                self._handler = None
                self._sequence_tracker.reset()
                self._timestamp_tracker.reset()
                self._expected_source = None
                self._expected_ssrc = None
                if not self._recovery_active:
                    self._audio_recovery_state = "stopped"

    def _receiver_loop(self) -> None:
        while not self._stop.is_set():
            with self._state_lock:
                rtp_socket = self._rtp_socket
            if rtp_socket is None:
                return
            try:
                datagram, source = rtp_socket.recvfrom(MAX_RTP_DATAGRAM_SIZE)
            except TimeoutError:
                continue
            except OSError:
                if not self._stop.is_set():
                    with self._statistics_lock:
                        self._statistics.receive_errors += 1
                    logger.exception("SDS200 RTP socket failed for %s", self.endpoint)
                    self._request_audio_recovery("rtp_receive_error", None)
                return
            if self._stop.is_set() or not datagram:
                continue
            with self._statistics_lock:
                self._statistics.datagrams_received += 1
                self._statistics.bytes_received += len(datagram)
            if source != self._expected_source:
                with self._statistics_lock:
                    self._statistics.unexpected_source_packets += 1
                logger.warning(
                    "Discarding SDS200 RTP packet from unexpected source %s:%s",
                    source[0],
                    source[1],
                )
                continue
            try:
                packet = RtpPacket.parse(datagram)
            except RtpProtocolError:
                with self._statistics_lock:
                    self._statistics.malformed_packets += 1
                logger.warning("Discarding invalid SDS200 RTP packet", exc_info=True)
                continue

            if self._expected_ssrc is None:
                self._expected_ssrc = packet.ssrc
                with self._statistics_lock:
                    self._statistics.ssrc = packet.ssrc
            elif packet.ssrc != self._expected_ssrc:
                with self._statistics_lock:
                    self._statistics.ssrc_mismatch_packets += 1
                logger.warning(
                    "Discarding SDS200 RTP packet with unexpected SSRC %s",
                    packet.ssrc,
                )
                continue

            observation = self._sequence_tracker.observe(packet.sequence)
            if observation.missing:
                with self._statistics_lock:
                    self._statistics.sequence_gaps += 1
                    self._statistics.packets_lost += observation.missing
                logger.warning(
                    "SDS200 RTP sequence gap: expected %s, received %s (%s missing)",
                    observation.expected,
                    observation.sequence,
                    observation.missing,
                )
            elif observation.duplicate:
                with self._statistics_lock:
                    self._statistics.duplicate_packets += 1
                logger.debug("Discarding duplicate SDS200 RTP packet %s", packet.sequence)
                continue
            elif observation.out_of_order:
                with self._statistics_lock:
                    self._statistics.late_packets += 1
                logger.debug("Discarding late SDS200 RTP packet %s", packet.sequence)
                continue

            timestamp = self._timestamp_tracker.observe(
                packet.timestamp,
                len(packet.payload),
            )
            if timestamp.missing_samples:
                with self._statistics_lock:
                    self._statistics.timestamp_discontinuities += 1
                    self._statistics.timestamp_samples_missing += (
                        timestamp.missing_samples
                    )
                logger.warning(
                    "SDS200 RTP timestamp discontinuity: expected %s, received %s "
                    "(%s samples missing)",
                    timestamp.expected,
                    timestamp.timestamp,
                    timestamp.missing_samples,
                )
            elif timestamp.backwards:
                with self._statistics_lock:
                    self._statistics.timestamp_discontinuities += 1
                    self._statistics.timestamp_backwards += 1
                logger.warning(
                    "SDS200 RTP timestamp moved backwards: expected %s, received %s",
                    timestamp.expected,
                    timestamp.timestamp,
                )

            with self._statistics_lock:
                statistics = self._statistics
                statistics.packets_delivered += 1
                statistics.payload_bytes_delivered += len(packet.payload)
                if statistics.first_sequence is None:
                    statistics.first_sequence = packet.sequence
                statistics.last_sequence = packet.sequence
                statistics.last_timestamp = packet.timestamp
            with self._state_lock:
                self._last_rtp_packet_monotonic = monotonic()
                self._rtp_watchdog_started_monotonic = None
                self._rtp_packet_seen_this_session = True
                self._rtp_ever_received = True

            observed_at = datetime.now(UTC)
            self.events.emit(
                "packet",
                PcmuPacket(
                    endpoint=self.endpoint,
                    sequence=packet.sequence,
                    timestamp=packet.timestamp,
                    ssrc=packet.ssrc,
                    payload=packet.payload,
                    observed_at=observed_at,
                    marker=packet.marker,
                    expected_sequence=observation.expected,
                    missing_packets=observation.missing,
                    expected_timestamp=timestamp.expected,
                    missing_samples=timestamp.missing_samples,
                    timestamp_backwards=timestamp.backwards,
                ),
            )

            handler = self._handler
            if handler is not None:
                try:
                    handler(
                        AudioChunk(
                            packet.payload,
                            received_at=observed_at,
                        )
                    )
                except Exception:
                    with self._statistics_lock:
                        self._statistics.callback_errors += 1
                    logger.exception("Unhandled exception in audio chunk callback")

    def _keepalive_loop(self) -> None:
        while not self._stop.wait(self.keepalive_interval):
            with self._state_lock:
                rtsp_client = self._rtsp_client
            if rtsp_client is None:
                return
            try:
                with self._rtsp_lock:
                    rtsp_client.get_parameter()
                with self._statistics_lock:
                    self._statistics.keepalives_sent += 1
                with self._state_lock:
                    self._last_keepalive_monotonic = monotonic()
                if self._rtp_has_been_inactive():
                    self._request_audio_recovery("rtp_inactive", None)
                    return
            except (OSError, RtspProtocolError, ScannerConnectionError):
                if not self._stop.is_set():
                    with self._statistics_lock:
                        self._statistics.keepalive_failures += 1
                    logger.exception("SDS200 RTSP keepalive failed for %s", self.endpoint)
                    self._request_audio_recovery("rtsp_keepalive_failure", None)
                return

    def _rtp_has_been_inactive(self) -> bool:
        with self._state_lock:
            last_packet = (
                self._last_rtp_packet_monotonic
                or self._rtp_watchdog_started_monotonic
            )
        return (
            last_packet is not None
            and monotonic() - last_packet >= self.inactivity_timeout
        )

    def _request_audio_recovery(
        self,
        reason: str,
        error: BaseException | None,
    ) -> None:
        with self._state_lock:
            if self._stop.is_set() or self._recovery_active:
                return
            self._recovery_active = True
            self._audio_recovery_count += 1
            self._audio_recovery_state = "recovering"
            self._last_audio_recovery_error = (
                error.__class__.__name__ if error is not None else reason
            )
            handler = self._handler
            recovery_thread = threading.Thread(
                target=self._recover_audio,
                args=(handler, reason),
                name="sds200-audio-recovery",
                daemon=True,
            )
            self._recovery_thread = recovery_thread
        logger.warning(
            "SDS200 network audio recovery requested endpoint=%s reason=%s",
            self.endpoint,
            reason,
        )
        recovery_thread.start()

    def _recover_audio(
        self,
        handler: AudioChunkHandler | None,
        reason: str,
    ) -> None:
        failure: BaseException | None = None
        if handler is None:
            failure = ScannerConnectionError("Audio recovery has no active handler.")
        else:
            for attempt in range(1, self.recovery_attempts + 1):
                if attempt > 1 and self._stop.wait(
                    min(self.recovery_backoff * (2 ** (attempt - 2)), 10.0)
                ):
                    self._finish_audio_recovery()
                    return
                if self._stop.is_set():
                    self._finish_audio_recovery()
                    return
                try:
                    with self._lifecycle_lock:
                        if self._stop.is_set():
                            self._finish_audio_recovery()
                            return
                        self.stop()
                        self._start_session(handler, reset_statistics=False)
                    self._finish_audio_recovery()
                    logger.info(
                        "SDS200 network audio recovery completed endpoint=%s "
                        "reason=%s attempt=%d",
                        self.endpoint,
                        reason,
                        attempt,
                    )
                    return
                except ScannerConnectionError as error:
                    failure = error
                    logger.warning(
                        "SDS200 network audio recovery failed endpoint=%s "
                        "reason=%s attempt=%d error=%s",
                        self.endpoint,
                        reason,
                        attempt,
                        error.__class__.__name__,
                    )
        with self._state_lock:
            self._audio_recovery_state = "failed"
            self._last_audio_recovery_error = (
                failure.__class__.__name__ if failure is not None else reason
            )
        self._finish_audio_recovery()
        logger.error(
            "SDS200 network audio recovery exhausted endpoint=%s reason=%s",
            self.endpoint,
            reason,
        )

    def _finish_audio_recovery(self) -> None:
        with self._state_lock:
            self._recovery_active = False
            self._recovery_thread = None
            if self._stop.is_set():
                self._audio_recovery_state = "stopped"
