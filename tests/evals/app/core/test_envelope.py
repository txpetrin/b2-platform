from dataclasses import fields
from datetime import date, datetime
import re

import pytest

from app.core.envelope import (
    CanonicalMessage,
    InputType,
    LogEvent,
    LocationContext,
    LocationSource,
    LogEventType,
    SubmissionType,
    _make_session_id_for_date,
    make_session_id,
)


# Enum value tests lock the serialized string contracts used across envelopes.
def test_input_type_values_are_lowercase_strings() -> None:
    assert InputType.TEXT == "text"
    assert InputType.IMAGE == "image"
    assert InputType.AUDIO == "audio"
    assert InputType.DOCUMENT == "document"
    assert InputType.LOCATION == "location"


def test_submission_type_values_are_lowercase_strings() -> None:
    assert SubmissionType.INTAKE == "intake"
    assert SubmissionType.VERIFICATION_EVIDENCE == "verification_evidence"
    assert SubmissionType.MONITORING_CHECKIN == "monitoring_checkin"
    assert SubmissionType.UNKNOWN == "unknown"


def test_location_source_values_are_lowercase_strings() -> None:
    assert LocationSource.DEVICE == "device"
    assert LocationSource.PROMPT == "prompt"
    assert LocationSource.PHONE_PREFIX == "phone_prefix"
    assert LocationSource.NONE == "none"


def test_log_event_type_values_are_lowercase_strings() -> None:
    assert LogEventType.MESSAGE_RECEIVED == "message_received"
    assert LogEventType.MESSAGE_NORMALIZED == "message_normalized"
    assert LogEventType.AGENT_PROMPT_CREATED == "agent_prompt_created"
    assert LogEventType.RESPONSE_SENT == "response_sent"
    assert LogEventType.ERROR == "error"


# Session ID tests cover deterministic hashing, daily rotation, and raw metadata privacy.
def test_session_id_is_consistent_within_a_day() -> None:
    day = date(2026, 5, 2)

    first = _make_session_id_for_date("user-123", "sms", day)
    second = _make_session_id_for_date("user-123", "sms", day)

    assert first == second


def test_session_id_rotates_daily() -> None:
    first = _make_session_id_for_date("user-123", "sms", date(2026, 5, 2))
    second = _make_session_id_for_date("user-123", "sms", date(2026, 5, 3))

    assert first != second


def test_session_id_differs_by_channel() -> None:
    sms = _make_session_id_for_date("user-123", "sms", date(2026, 5, 2))
    whatsapp = _make_session_id_for_date("user-123", "whatsapp", date(2026, 5, 2))

    assert sms != whatsapp


def test_session_id_is_sha256_hex_digest() -> None:
    session_id = _make_session_id_for_date("user-123", "sms", date(2026, 5, 2))

    assert len(session_id) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", session_id)


def test_make_session_id_returns_sha256_hex_digest() -> None:
    session_id = make_session_id("user-123", "sms")

    assert len(session_id) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", session_id)


def test_session_id_does_not_include_raw_user_id() -> None:
    raw_user_id = "user-123"

    session_id = _make_session_id_for_date(raw_user_id, "sms", date(2026, 5, 2))

    assert raw_user_id not in session_id


# Location tests keep approximate context useful while preventing coordinate leakage.
def test_location_context_defaults() -> None:
    context = LocationContext()

    assert context.country_code is None
    assert context.region is None
    assert context.city is None
    assert context.source is LocationSource.NONE
    assert context.confidence == 0.0


def test_location_context_accepts_valid_confidence() -> None:
    context = LocationContext(
        country_code="US",
        region="CA",
        city="San Francisco",
        source=LocationSource.DEVICE,
        confidence=0.75,
    )

    assert context.confidence == 0.75


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_location_context_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0 inclusive"):
        LocationContext(confidence=confidence)


def test_location_context_has_no_coordinate_fields() -> None:
    field_names = {field.name for field in fields(LocationContext)}

    forbidden = {"lat", "lng", "latitude", "longitude", "coordinates", "gps", "raw_location"}
    assert field_names.isdisjoint(forbidden)


def test_location_context_safe_dict_excludes_coordinate_keys() -> None:
    payload = LocationContext(source=LocationSource.PROMPT, confidence=1.0).to_safe_dict()

    assert payload == {
        "country_code": None,
        "region": None,
        "city": None,
        "source": LocationSource.PROMPT,
        "confidence": 1.0,
    }
    forbidden = {"lat", "lng", "latitude", "longitude", "coordinates", "gps", "raw_location"}
    assert set(payload).isdisjoint(forbidden)


def test_canonical_message_defaults_to_unknown_submission_type() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    assert message.submission_type is SubmissionType.UNKNOWN


def test_canonical_message_session_context_does_not_share_mutable_state() -> None:
    first = CanonicalMessage(session_id="session-1", channel="sms", input_type=InputType.TEXT)
    second = CanonicalMessage(session_id="session-2", channel="sms", input_type=InputType.TEXT)

    first.session_context["key"] = "value"

    assert second.session_context == {}


def test_canonical_message_prior_context_does_not_share_mutable_state() -> None:
    first = CanonicalMessage(session_id="session-1", channel="sms", input_type=InputType.TEXT)
    second = CanonicalMessage(session_id="session-2", channel="sms", input_type=InputType.TEXT)

    first.prior_context.append({"role": "user"})

    assert second.prior_context == []


def test_canonical_message_timestamp_is_timezone_aware() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    assert message.timestamp.tzinfo is not None
    assert message.timestamp.utcoffset() is not None
    assert datetime.now(message.timestamp.tzinfo).utcoffset() == message.timestamp.utcoffset()


# Agent prompt tests verify only privacy-safe fields are serialized for runtime use.
def test_agent_prompt_includes_text_content_for_text_messages() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        text_content="I need help with an application.",
    )

    prompt = message.to_agent_prompt()

    assert prompt["text_content"] == "I need help with an application."


def test_agent_prompt_excludes_session_id() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    prompt = message.to_agent_prompt()

    assert "session_id" not in prompt


def test_agent_prompt_excludes_channel() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    prompt = message.to_agent_prompt()

    assert "channel" not in prompt


def test_agent_prompt_excludes_media_url() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.IMAGE,
        media_url="https://example.test/image.jpg",
    )

    prompt = message.to_agent_prompt()

    assert "media_url" not in prompt


def test_agent_prompt_drops_text_content_containing_phone_number() -> None:
    # Defense-in-depth: if PII scrubbing failed open and a phone slipped through,
    # to_agent_prompt() must not forward it to the LLM.
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        text_content="Call me at +1 555 010 1234 for details.",
    )

    prompt = message.to_agent_prompt()

    assert "text_content" not in prompt


def test_agent_prompt_includes_clean_text_content_without_phone_number() -> None:
    # Scrubbed text (phone replaced with placeholder) must still pass through.
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        text_content="Call me at [PHONE_NUMBER_3f8a2b1c] for details.",
    )

    prompt = message.to_agent_prompt()

    assert prompt["text_content"] == "Call me at [PHONE_NUMBER_3f8a2b1c] for details."


def test_agent_prompt_includes_safe_location_fields_when_location_context_exists() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        location_context=LocationContext(
            country_code="US",
            region="CA",
            city="Oakland",
            source=LocationSource.PROMPT,
            confidence=0.8,
        ),
    )

    prompt = message.to_agent_prompt()

    assert prompt["location_context"] == {
        "country_code": "US",
        "region": "CA",
        "city": "Oakland",
        "source": LocationSource.PROMPT,
        "confidence": 0.8,
    }


def test_agent_prompt_excludes_coordinate_like_keys() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        location_context=LocationContext(
            country_code="US",
            source=LocationSource.PROMPT,
            confidence=0.9,
        ),
        session_context={"lat": 37.8, "safe": "value"},
    )

    prompt = message.to_agent_prompt()
    forbidden = {"coordinates", "latitude", "longitude", "lat", "lng", "gps"}

    assert set(prompt).isdisjoint(forbidden)
    assert set(prompt["location_context"]).isdisjoint(forbidden)
    assert "session_context" not in prompt


def test_agent_prompt_handles_missing_location_context_cleanly() -> None:
    message = CanonicalMessage(session_id="session-123", channel="sms", input_type=InputType.TEXT)

    prompt = message.to_agent_prompt()

    assert "location_context" not in prompt


def test_agent_prompt_does_not_mutate_original_message() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        session_context={"eligible_programs": ["housing"]},
    )

    prompt = message.to_agent_prompt()
    prompt["session_context"]["eligible_programs"].append("food")

    assert message.session_context == {"eligible_programs": ["housing"]}


# Log event tests cover safe cleartext metadata and opaque PII envelope handling.
def test_log_event_default_values() -> None:
    event = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
    )

    assert event.cleartext_payload == {}
    assert event.pii_envelope is None
    assert event.agent_version is None
    assert event.platform_version == "0.1.0"
    assert event.schema_version == "1.0"


def test_log_event_timestamp_is_timezone_aware() -> None:
    event = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
    )

    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset() is not None
    assert datetime.now(event.timestamp.tzinfo).utcoffset() == event.timestamp.utcoffset()


@pytest.mark.parametrize("pii_envelope", [b"opaque-bytes", None])
def test_log_event_pii_envelope_can_be_bytes_or_none(pii_envelope: bytes | None) -> None:
    event = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
        pii_envelope=pii_envelope,
    )

    assert event.pii_envelope == pii_envelope


@pytest.mark.parametrize(
    "key",
    [
        "phone",
        "phone_number",
        "raw_phone",
        "name",
        "full_name",
        "lat",
        "lng",
        "latitude",
        "longitude",
        "coordinates",
        "gps",
        "raw_location",
    ],
)
def test_log_event_forbidden_cleartext_payload_keys_raise_value_error(key: str) -> None:
    with pytest.raises(ValueError, match="cleartext_payload contains forbidden PII keys"):
        LogEvent(
            event_type=LogEventType.MESSAGE_RECEIVED,
            session_id_hash="session-hash",
            cleartext_payload={"nested": {key: "unsafe"}},
        )


def test_log_event_safe_cleartext_payload_is_accepted() -> None:
    event = LogEvent(
        event_type=LogEventType.AGENT_PROMPT_CREATED,
        session_id_hash="session-hash",
        cleartext_payload={
            "input_type": InputType.TEXT,
            "submission_type": SubmissionType.INTAKE,
            "location_context": {
                "country_code": "US",
                "region": "CA",
                "city": "Oakland",
                "source": LocationSource.PROMPT,
                "confidence": 0.8,
            },
        },
        pii_envelope=b"encrypted-user-data",
    )

    assert event.cleartext_payload["input_type"] == InputType.TEXT
    assert event.pii_envelope == b"encrypted-user-data"


# Emit tests intentionally use a tiny fake session instead of introducing session architecture.
class FakeSession:
    def __init__(self) -> None:
        self.log_buffer: list[LogEvent] = []


def test_log_event_emit_appends_event_to_session_log_buffer() -> None:
    session = FakeSession()
    event = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
    )

    event.emit(session)

    assert session.log_buffer == [event]


def test_log_event_multiple_emits_preserve_order() -> None:
    session = FakeSession()
    first = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
    )
    second = LogEvent(
        event_type=LogEventType.MESSAGE_NORMALIZED,
        session_id_hash="session-hash",
    )

    first.emit(session)
    second.emit(session)

    assert session.log_buffer == [first, second]


def test_log_event_emit_missing_log_buffer_raises_attribute_error() -> None:
    event = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
    )

    with pytest.raises(AttributeError, match="session must expose a log_buffer attribute"):
        event.emit(object())


def test_log_event_emit_non_appendable_log_buffer_raises_clear_error() -> None:
    class BadSession:
        log_buffer = object()

    event = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
    )

    with pytest.raises(AttributeError, match="session.log_buffer must be list-like and expose append"):
        event.emit(BadSession())


def test_log_event_emit_introduces_no_external_side_effects() -> None:
    session = FakeSession()
    event = LogEvent(
        event_type=LogEventType.MESSAGE_RECEIVED,
        session_id_hash="session-hash",
        cleartext_payload={"status": "received"},
    )

    before_payload = dict(event.cleartext_payload)
    event.emit(session)

    assert session.log_buffer == [event]
    assert event.cleartext_payload == before_payload
    assert event.pii_envelope is None


# Privacy regression tests pin sensitive exclusions so future edits do not loosen them.
def test_privacy_regression_location_context_has_no_coordinate_fields() -> None:
    field_names = {field.name for field in fields(LocationContext)}
    forbidden = {"lat", "lng", "latitude", "longitude", "coordinates", "gps", "raw_location"}

    assert field_names.isdisjoint(forbidden)


def test_privacy_regression_agent_prompt_excludes_private_and_location_keys() -> None:
    message = CanonicalMessage(
        session_id="session-123",
        channel="sms",
        input_type=InputType.TEXT,
        media_url="https://example.test/file.jpg",
        session_context={
            "phone": "+1 555 010 1234",
            "phone_number": "+1 555 010 5678",
            "raw_phone": "+1 555 010 9999",
            "coordinates": [37.8, -122.3],
        },
        prior_context=[
            {
                "session_id": "session-123",
                "channel": "sms",
            },
        ],
    )

    prompt = message.to_agent_prompt()
    forbidden = {
        "session_id",
        "channel",
        "media_url",
        "phone",
        "phone_number",
        "raw_phone",
        "coordinates",
    }

    assert _collect_keys(prompt).isdisjoint(forbidden)


@pytest.mark.parametrize(
    "key",
    [
        "phone",
        "phone_number",
        "raw_phone",
        "name",
        "full_name",
        "lat",
        "lng",
        "latitude",
        "longitude",
        "coordinates",
        "gps",
        "raw_location",
    ],
)
def test_privacy_regression_log_event_rejects_cleartext_pii_location_keys(key: str) -> None:
    with pytest.raises(ValueError, match="cleartext_payload contains forbidden PII keys"):
        LogEvent(
            event_type=LogEventType.MESSAGE_RECEIVED,
            session_id_hash="session-hash",
            cleartext_payload={"event": {"metadata": {key: "unsafe"}}},
        )


def test_privacy_regression_make_session_id_does_not_expose_raw_channel_user_id() -> None:
    raw_channel_user_id = "raw-user-123"

    session_id = make_session_id(raw_channel_user_id, "sms")

    assert raw_channel_user_id not in session_id


def test_privacy_regression_make_session_id_does_not_expose_raw_channel_name() -> None:
    raw_channel = "sms"

    session_id = make_session_id("raw-user-123", raw_channel)

    assert raw_channel not in session_id


def _collect_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = {str(key) for key in value}
        for child in value.values():
            keys.update(_collect_keys(child))
        return keys

    if isinstance(value, list | tuple):
        keys: set[str] = set()
        for item in value:
            keys.update(_collect_keys(item))
        return keys

    return set()
