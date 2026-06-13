"""Tests for app/core/pii.py — PII scrubbing layer.

Tests are organised into five sections:
1. Configuration loading (PiiConfig, EntityConfig)
2. Data types (PiiEntity, ScrubResult)
3. PiiAuditStore — isolation, daily rotation, file persistence
4. PiiScrubber — scrubbing logic with Presidio mocked
5. Retry loop — failure classification, recovery, fail-open behaviour
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.envelope import (
    CanonicalMessage,
    InputType,
    LocationContext,
    LocationSource,
    SubmissionType,
    make_session_id,
)
from app.core.pii import (
    AuditStoreConfig,
    EntityConfig,
    NlpEngineConfig,
    PiiAuditStore,
    PiiConfig,
    PiiEntity,
    PiiScrubber,
    ScrubResult,
    _classify_failure,
    _FailureKind,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_config() -> PiiConfig:
    """Regex-only Phase 1 config with no NER entities."""
    return PiiConfig(
        score_threshold=0.5,
        max_retries=3,
        retry_timeout_ms=200,
        language="en",
        entities=[
            EntityConfig(name="PHONE_NUMBER", enabled=True, recognizer="regex"),
            EntityConfig(name="EMAIL_ADDRESS", enabled=True, recognizer="regex"),
            EntityConfig(name="CREDIT_CARD", enabled=True, recognizer="regex"),
            EntityConfig(name="IP_ADDRESS", enabled=True, recognizer="regex"),
        ],
        nlp_engine=NlpEngineConfig(provider="spacy", model_name="en_core_web_sm"),
        audit_store=AuditStoreConfig(
            memory_ttl_hours=24,
            file_path="data/test_pii_audit.jsonl",
            store_originals=True,
        ),
    )


@pytest.fixture()
def sample_message() -> CanonicalMessage:
    return CanonicalMessage(
        session_id=make_session_id("user-001", "whatsapp"),
        channel="whatsapp",
        input_type=InputType.TEXT,
        text_content="Hello, my number is +1 555 010 1234 and my email is test@example.com",
    )


@pytest.fixture()
def clean_message() -> CanonicalMessage:
    return CanonicalMessage(
        session_id=make_session_id("user-002", "sms"),
        channel="sms",
        input_type=InputType.TEXT,
        text_content="My maize in Kisumu is turning yellow. It started last Tuesday.",
    )


@pytest.fixture()
def mock_analyzer():
    """A Presidio AnalyzerEngine mock that returns a configurable list of results."""
    analyzer = MagicMock()
    analyzer.analyze.return_value = []
    return analyzer


@pytest.fixture()
def scrubber_with_mock(minimal_config: PiiConfig, mock_analyzer: MagicMock) -> PiiScrubber:
    """PiiScrubber with Presidio's AnalyzerEngine replaced by a mock."""
    scrubber = PiiScrubber.__new__(PiiScrubber)
    scrubber._config = minimal_config
    scrubber._analyzer = mock_analyzer
    return scrubber


@pytest.fixture()
def tmp_audit_store(tmp_path: Path, minimal_config: PiiConfig) -> PiiAuditStore:
    cfg = AuditStoreConfig(
        memory_ttl_hours=24,
        file_path=str(tmp_path / "test_audit.jsonl"),
        store_originals=True,
    )
    return PiiAuditStore(cfg)


# ---------------------------------------------------------------------------
# 1. Configuration loading
# ---------------------------------------------------------------------------

class TestPiiConfig:
    def test_from_yaml_loads_phase1_entities(self) -> None:
        config = PiiConfig.from_yaml()
        enabled = config.enabled_entity_names
        assert "PHONE_NUMBER" in enabled
        assert "EMAIL_ADDRESS" in enabled
        assert "CREDIT_CARD" in enabled
        assert "IBAN_CODE" in enabled
        assert "IP_ADDRESS" in enabled

    def test_from_yaml_phase2_entities_disabled(self) -> None:
        config = PiiConfig.from_yaml()
        disabled = [e for e in config.entities if not e.enabled]
        disabled_names = {e.name for e in disabled}
        # PERSON and LOCATION are Phase 2
        assert "PERSON" in disabled_names
        assert "LOCATION" in disabled_names

    def test_from_yaml_needs_ner_is_false_for_phase1(self) -> None:
        config = PiiConfig.from_yaml()
        assert config.needs_ner is False

    def test_needs_ner_true_when_ner_entity_enabled(self) -> None:
        config = PiiConfig(
            entities=[
                EntityConfig(name="PERSON", enabled=True, recognizer="ner"),
            ]
        )
        assert config.needs_ner is True

    def test_enabled_entity_names_excludes_disabled(self) -> None:
        config = PiiConfig(
            entities=[
                EntityConfig(name="PHONE_NUMBER", enabled=True, recognizer="regex"),
                EntityConfig(name="PERSON", enabled=False, recognizer="ner"),
            ]
        )
        assert config.enabled_entity_names == ["PHONE_NUMBER"]

    def test_entity_config_allow_list_is_tuple(self) -> None:
        ec = EntityConfig(
            name="LOCATION",
            enabled=False,
            recognizer="ner",
            allow_list=("Kenya", "Nairobi"),
        )
        assert isinstance(ec.allow_list, tuple)
        assert "Kenya" in ec.allow_list

    def test_audit_store_config_store_originals_default_true(self) -> None:
        config = PiiConfig.from_yaml()
        assert config.audit_store.store_originals is True

    def test_scrubber_retry_timeout_positive(self) -> None:
        config = PiiConfig.from_yaml()
        assert config.retry_timeout_ms > 0
        assert config.max_retries > 0


# ---------------------------------------------------------------------------
# 2. Data types
# ---------------------------------------------------------------------------

class TestPiiEntity:
    def test_frozen(self) -> None:
        entity = PiiEntity(
            field_path="text_content",
            entity_type="PHONE_NUMBER",
            original_value="+1 555 010 1234",
            sha256_hash="abc123",
            placeholder="[PHONE_NUMBER_abc123]",
            score=0.85,
            start=20,
            end=35,
        )
        with pytest.raises(AttributeError):
            entity.original_value = "changed"  # type: ignore[misc]

    def test_sha256_hash_not_equal_to_original(self) -> None:
        import hashlib
        original = "+254712345678"
        h = hashlib.sha256(original.encode()).hexdigest()
        entity = PiiEntity(
            field_path="text_content",
            entity_type="PHONE_NUMBER",
            original_value=original,
            sha256_hash=h,
            placeholder=f"[PHONE_NUMBER_{h[:8]}]",
            score=0.85,
            start=0,
            end=len(original),
        )
        assert entity.sha256_hash != entity.original_value
        assert len(entity.sha256_hash) == 64


class TestScrubResult:
    def test_found_pii_false_when_no_entities(self, sample_message: CanonicalMessage) -> None:
        result = ScrubResult(clean_message=sample_message, entities=[])
        assert result.found_pii is False

    def test_found_pii_true_when_entities_present(
        self, sample_message: CanonicalMessage
    ) -> None:
        entity = PiiEntity(
            field_path="text_content",
            entity_type="PHONE_NUMBER",
            original_value="+1 555 010 1234",
            sha256_hash="a" * 64,
            placeholder="[PHONE_NUMBER_aaaaaaaa]",
            score=0.85,
            start=0,
            end=15,
        )
        result = ScrubResult(clean_message=sample_message, entities=[entity])
        assert result.found_pii is True

    def test_scrub_failed_defaults_false(self, sample_message: CanonicalMessage) -> None:
        result = ScrubResult(clean_message=sample_message, entities=[])
        assert result.scrub_failed is False

    def test_scrubbed_at_is_utc(self, sample_message: CanonicalMessage) -> None:
        result = ScrubResult(clean_message=sample_message, entities=[])
        assert result.scrubbed_at.tzinfo is not None

    def test_clean_message_differs_from_raw_when_pii_found(
        self, scrubber_with_mock: PiiScrubber, sample_message: CanonicalMessage
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult(entity_type="PHONE_NUMBER", start=20, end=35, score=0.85)
        ]
        result = scrubber_with_mock.scrub(sample_message)
        assert result.clean_message.text_content != sample_message.text_content
        assert result.found_pii is True


# ---------------------------------------------------------------------------
# 3. PiiAuditStore
# ---------------------------------------------------------------------------

class TestPiiAuditStore:
    def test_record_appends_to_memory(
        self,
        tmp_audit_store: PiiAuditStore,
        sample_message: CanonicalMessage,
    ) -> None:
        result = ScrubResult(clean_message=sample_message, entities=[])
        tmp_audit_store.record(sample_message.session_id, result)
        assert tmp_audit_store.today_count() == 1

    def test_record_writes_to_file(
        self,
        tmp_audit_store: PiiAuditStore,
        sample_message: CanonicalMessage,
        tmp_path: Path,
    ) -> None:
        result = ScrubResult(clean_message=sample_message, entities=[])
        tmp_audit_store.record(sample_message.session_id, result)

        file_path = Path(tmp_audit_store._config.file_path)
        lines = file_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["session_id_hash"] == sample_message.session_id

    def test_store_originals_true_includes_original_value(
        self,
        tmp_audit_store: PiiAuditStore,
        sample_message: CanonicalMessage,
    ) -> None:
        entity = PiiEntity(
            field_path="text_content",
            entity_type="PHONE_NUMBER",
            original_value="+1 555 010 1234",
            sha256_hash="a" * 64,
            placeholder="[PHONE_NUMBER_aaaaaaaa]",
            score=0.85,
            start=0,
            end=15,
        )
        result = ScrubResult(clean_message=sample_message, entities=[entity])
        tmp_audit_store.record(sample_message.session_id, result)

        file_path = Path(tmp_audit_store._config.file_path)
        data = json.loads(file_path.read_text(encoding="utf-8").strip())
        assert data["entities"][0]["original_value"] == "+1 555 010 1234"

    def test_store_originals_false_excludes_original_value(
        self,
        tmp_path: Path,
        sample_message: CanonicalMessage,
    ) -> None:
        cfg = AuditStoreConfig(
            memory_ttl_hours=24,
            file_path=str(tmp_path / "no_originals.jsonl"),
            store_originals=False,
        )
        store = PiiAuditStore(cfg)
        entity = PiiEntity(
            field_path="text_content",
            entity_type="PHONE_NUMBER",
            original_value="+1 555 010 1234",
            sha256_hash="b" * 64,
            placeholder="[PHONE_NUMBER_bbbbbbbb]",
            score=0.85,
            start=0,
            end=15,
        )
        result = ScrubResult(clean_message=sample_message, entities=[entity])
        store.record(sample_message.session_id, result)

        file_path = Path(cfg.file_path)
        data = json.loads(file_path.read_text(encoding="utf-8").strip())
        assert "original_value" not in data["entities"][0]
        assert "sha256_hash" in data["entities"][0]

    def test_daily_rotation_clears_memory(
        self,
        tmp_audit_store: PiiAuditStore,
        sample_message: CanonicalMessage,
    ) -> None:
        result = ScrubResult(clean_message=sample_message, entities=[])
        tmp_audit_store.record(sample_message.session_id, result)
        assert tmp_audit_store.today_count() == 1

        # Simulate day change
        from datetime import timedelta
        tmp_audit_store._current_day = date(2000, 1, 1)
        assert tmp_audit_store.today_count() == 0

    def test_scrub_failed_recorded_in_file(
        self,
        tmp_audit_store: PiiAuditStore,
        sample_message: CanonicalMessage,
    ) -> None:
        result = ScrubResult(
            clean_message=sample_message,
            entities=[],
            scrub_failed=True,
            failure_reason="encoding: bad bytes",
        )
        tmp_audit_store.record(sample_message.session_id, result)

        file_path = Path(tmp_audit_store._config.file_path)
        data = json.loads(file_path.read_text(encoding="utf-8").strip())
        assert data["scrub_failed"] is True
        assert data["failure_reason"] == "encoding: bad bytes"

    def test_file_write_failure_does_not_raise(
        self,
        sample_message: CanonicalMessage,
        tmp_path: Path,
    ) -> None:
        cfg = AuditStoreConfig(
            memory_ttl_hours=24,
            file_path="/nonexistent_root/cant_write.jsonl",
            store_originals=True,
        )
        store = PiiAuditStore.__new__(PiiAuditStore)
        store._config = cfg
        store._memory = []
        store._current_day = datetime.now(timezone.utc).date()
        store._file_path = Path(cfg.file_path)

        result = ScrubResult(clean_message=sample_message, entities=[])
        # Should not raise — failure is logged only
        store._append_to_file(
            store._memory  # wrong type on purpose; we just want no raise
        )


# ---------------------------------------------------------------------------
# 4. PiiScrubber — scrubbing logic
# ---------------------------------------------------------------------------

class TestPiiScrubberScrubbing:
    def test_no_pii_returns_identical_text(
        self,
        scrubber_with_mock: PiiScrubber,
        clean_message: CanonicalMessage,
    ) -> None:
        result = scrubber_with_mock.scrub(clean_message)
        assert result.clean_message.text_content == clean_message.text_content
        assert not result.found_pii
        assert not result.scrub_failed

    def test_phone_replaced_in_text_content(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult(entity_type="PHONE_NUMBER", start=20, end=35, score=0.85)
        ]
        result = scrubber_with_mock.scrub(sample_message)
        assert "PHONE_NUMBER" in result.clean_message.text_content
        assert "+1 555 010 1234" not in result.clean_message.text_content
        assert result.found_pii

    def test_original_preserved_in_entity(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        phone = "+1 555 010 1234"
        start = sample_message.text_content.index(phone)
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=start,
                end=start + len(phone),
                score=0.85,
            )
        ]
        result = scrubber_with_mock.scrub(sample_message)
        assert result.entities[0].original_value == phone

    def test_entity_sha256_hash_matches_original(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        import hashlib
        from presidio_analyzer import RecognizerResult
        phone = "+1 555 010 1234"
        start = sample_message.text_content.index(phone)
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult(
                entity_type="PHONE_NUMBER",
                start=start,
                end=start + len(phone),
                score=0.85,
            )
        ]
        result = scrubber_with_mock.scrub(sample_message)
        expected_hash = hashlib.sha256(phone.encode()).hexdigest()
        assert result.entities[0].sha256_hash == expected_hash

    def test_field_path_is_text_content(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult(entity_type="EMAIL_ADDRESS", start=47, end=63, score=0.9)
        ]
        result = scrubber_with_mock.scrub(sample_message)
        assert result.entities[0].field_path == "text_content"

    def test_pii_in_session_context_stripped(
        self,
        scrubber_with_mock: PiiScrubber,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        message = CanonicalMessage(
            session_id=make_session_id("user-003", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content="safe text",
            session_context={"contact": "+254712345678", "region": "Kisumu"},
        )
        # Mock: first call (text_content) returns nothing; second call (session_context.contact) returns phone
        scrubber_with_mock._analyzer.analyze.side_effect = [
            [],                                                    # text_content
            [RecognizerResult("PHONE_NUMBER", 0, 13, 0.9)],       # session_context.contact
            [],                                                    # session_context.region
        ]
        result = scrubber_with_mock.scrub(message)
        assert "PHONE_NUMBER" in result.clean_message.session_context["contact"]
        assert result.clean_message.session_context["region"] == "Kisumu"
        assert result.entities[0].field_path == "session_context.contact"

    def test_pii_in_prior_context_stripped(
        self,
        scrubber_with_mock: PiiScrubber,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        message = CanonicalMessage(
            session_id=make_session_id("user-004", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content="",
            prior_context=[{"role": "user", "content": "my email is me@test.com"}],
        )
        scrubber_with_mock._analyzer.analyze.side_effect = [
            [],                                                        # text_content
            [],                                                        # prior_context[0].role
            [RecognizerResult("EMAIL_ADDRESS", 12, 23, 0.9)],         # prior_context[0].content
        ]
        result = scrubber_with_mock.scrub(message)
        assert "EMAIL_ADDRESS" in result.clean_message.prior_context[0]["content"]
        assert result.entities[0].field_path == "prior_context[0].content"

    def test_original_message_not_mutated(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult(entity_type="PHONE_NUMBER", start=20, end=35, score=0.85)
        ]
        original_text = sample_message.text_content
        scrubber_with_mock.scrub(sample_message)
        assert sample_message.text_content == original_text

    def test_clean_message_is_new_instance(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult(entity_type="PHONE_NUMBER", start=20, end=35, score=0.85)
        ]
        result = scrubber_with_mock.scrub(sample_message)
        assert result.clean_message is not sample_message

    def test_allow_list_prevents_flagging(
        self,
        minimal_config: PiiConfig,
        mock_analyzer: MagicMock,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        config = PiiConfig(
            entities=[
                EntityConfig(
                    name="LOCATION",
                    enabled=True,
                    recognizer="ner",
                    allow_list=("Kenya", "Nairobi"),
                ),
            ],
            nlp_engine=minimal_config.nlp_engine,
            audit_store=minimal_config.audit_store,
        )
        scrubber = PiiScrubber.__new__(PiiScrubber)
        scrubber._config = config
        scrubber._analyzer = mock_analyzer

        message = CanonicalMessage(
            session_id=make_session_id("u", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content="I am in Nairobi, Kenya.",
        )
        # "I am in Nairobi, Kenya."
        #  01234567890123456789012
        #          ^8    ^15 ^17  ^22
        mock_analyzer.analyze.return_value = [
            RecognizerResult(entity_type="LOCATION", start=8, end=15, score=0.9),   # Nairobi
            RecognizerResult(entity_type="LOCATION", start=17, end=22, score=0.9),  # Kenya
        ]
        result = scrubber.scrub(message)
        # Both values are in allow_list → no PII stripped
        assert result.clean_message.text_content == message.text_content
        assert not result.found_pii

    def test_none_text_content_not_processed(
        self,
        scrubber_with_mock: PiiScrubber,
    ) -> None:
        message = CanonicalMessage(
            session_id=make_session_id("u", "sms"),
            channel="sms",
            input_type=InputType.IMAGE,
            text_content=None,
        )
        result = scrubber_with_mock.scrub(message)
        assert result.clean_message.text_content is None
        scrubber_with_mock._analyzer.analyze.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Failure classification and retry loop
# ---------------------------------------------------------------------------

class TestFailureClassification:
    def test_unicode_decode_error_classified_as_encoding(self) -> None:
        exc = UnicodeDecodeError("utf-8", b"", 0, 1, "reason")
        assert _classify_failure(exc) == _FailureKind.ENCODING

    def test_memory_error_classified_correctly(self) -> None:
        assert _classify_failure(MemoryError()) == _FailureKind.MEMORY

    def test_timeout_error_classified_correctly(self) -> None:
        assert _classify_failure(TimeoutError()) == _FailureKind.TIMEOUT

    def test_model_not_found_classified_from_message(self) -> None:
        exc = RuntimeError("spacy model not found en_core_web_sm")
        assert _classify_failure(exc) == _FailureKind.MODEL_NOT_LOADED

    def test_unknown_exception_classified_as_unknown(self) -> None:
        assert _classify_failure(ValueError("something weird")) == _FailureKind.UNKNOWN


class TestRetryLoop:
    def test_succeeds_on_second_attempt(
        self,
        minimal_config: PiiConfig,
        sample_message: CanonicalMessage,
    ) -> None:
        scrubber = PiiScrubber.__new__(PiiScrubber)
        scrubber._config = minimal_config

        call_count = 0

        def flaky_do_scrub(msg: CanonicalMessage) -> ScrubResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return ScrubResult(clean_message=msg, entities=[])

        scrubber._do_scrub = flaky_do_scrub
        scrubber._recover = MagicMock()

        result = scrubber.scrub(sample_message)
        assert not result.scrub_failed
        assert call_count == 2
        scrubber._recover.assert_called_once()

    def test_fails_open_after_all_retries(
        self,
        minimal_config: PiiConfig,
        sample_message: CanonicalMessage,
    ) -> None:
        scrubber = PiiScrubber.__new__(PiiScrubber)
        scrubber._config = replace(minimal_config, max_retries=2)
        scrubber._do_scrub = MagicMock(side_effect=RuntimeError("persistent failure"))
        scrubber._recover = MagicMock()

        result = scrubber.scrub(sample_message)
        assert result.scrub_failed is True
        assert result.failure_reason is not None
        # text_content is nulled — unscanned PII must not reach the Agentic OS
        assert result.clean_message is not sample_message
        assert result.clean_message.text_content is None
        # Non-text fields are safe to preserve
        assert result.clean_message.session_id == sample_message.session_id
        assert result.clean_message.channel == sample_message.channel
        assert not result.found_pii

    def test_fail_open_nulls_text_content_to_block_unscanned_pii(
        self,
        minimal_config: PiiConfig,
        sample_message: CanonicalMessage,
    ) -> None:
        scrubber = PiiScrubber.__new__(PiiScrubber)
        scrubber._config = minimal_config
        scrubber._do_scrub = MagicMock(side_effect=MemoryError("OOM"))
        scrubber._recover = MagicMock()

        result = scrubber.scrub(sample_message)
        assert result.scrub_failed
        # text_content is nulled so unscanned PII cannot reach the Agentic OS
        assert result.clean_message.text_content is None
        # Structural fields that carry no free-text PII are preserved
        assert result.clean_message.input_type == sample_message.input_type
        assert result.clean_message.language_hint == sample_message.language_hint
        assert result.clean_message.location_context == sample_message.location_context

    def test_timeout_stops_retry_loop_early(
        self,
        minimal_config: PiiConfig,
        sample_message: CanonicalMessage,
    ) -> None:
        import time
        scrubber = PiiScrubber.__new__(PiiScrubber)
        # Set an extremely tight budget — 1ms
        scrubber._config = replace(minimal_config, max_retries=10, retry_timeout_ms=1)

        call_count = 0

        def slow_do_scrub(msg: CanonicalMessage) -> ScrubResult:
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # 10ms — exceeds 1ms budget
            raise RuntimeError("slow")

        scrubber._do_scrub = slow_do_scrub
        scrubber._recover = MagicMock()

        result = scrubber.scrub(sample_message)
        assert result.scrub_failed
        # Should not have attempted all 10 retries
        assert call_count < 10

    def test_model_reload_attempted_on_model_not_loaded(
        self,
        minimal_config: PiiConfig,
        sample_message: CanonicalMessage,
    ) -> None:
        scrubber = PiiScrubber.__new__(PiiScrubber)
        scrubber._config = replace(minimal_config, max_retries=2)
        scrubber._do_scrub = MagicMock(
            side_effect=RuntimeError("spacy model not found")
        )
        rebuild_called = []

        def mock_build() -> MagicMock:
            rebuild_called.append(True)
            return MagicMock()

        scrubber._build_analyzer = mock_build
        scrubber._analyzer = MagicMock()

        result = scrubber.scrub(sample_message)
        assert result.scrub_failed
        assert len(rebuild_called) >= 1


# ---------------------------------------------------------------------------
# Privacy regression: ScrubResult never bleeds into CanonicalMessage
# ---------------------------------------------------------------------------

class TestPrivacyIsolation:
    def test_entities_not_on_clean_message(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult("PHONE_NUMBER", 20, 35, 0.85)
        ]
        result = scrubber_with_mock.scrub(sample_message)
        # CanonicalMessage has no concept of entities — they live only in ScrubResult
        assert not hasattr(result.clean_message, "entities")

    def test_session_id_unchanged_in_clean_message(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult("PHONE_NUMBER", 20, 35, 0.85)
        ]
        result = scrubber_with_mock.scrub(sample_message)
        assert result.clean_message.session_id == sample_message.session_id

    def test_channel_unchanged_in_clean_message(
        self,
        scrubber_with_mock: PiiScrubber,
        sample_message: CanonicalMessage,
    ) -> None:
        from presidio_analyzer import RecognizerResult
        scrubber_with_mock._analyzer.analyze.return_value = [
            RecognizerResult("PHONE_NUMBER", 20, 35, 0.85)
        ]
        result = scrubber_with_mock.scrub(sample_message)
        assert result.clean_message.channel == sample_message.channel

    def test_to_agent_prompt_on_clean_message_excludes_pii_keys(
        self,
        scrubber_with_mock: PiiScrubber,
    ) -> None:
        """After scrubbing, to_agent_prompt() must still exclude all forbidden keys."""
        from presidio_analyzer import RecognizerResult
        message = CanonicalMessage(
            session_id=make_session_id("u", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content="safe text after scrub",
        )
        scrubber_with_mock._analyzer.analyze.return_value = []
        result = scrubber_with_mock.scrub(message)
        prompt = result.clean_message.to_agent_prompt()
        assert "session_id" not in prompt
        assert "channel" not in prompt
        assert "media_url" not in prompt
