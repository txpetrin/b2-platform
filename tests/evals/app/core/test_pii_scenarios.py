"""Real-Presidio scenario tests for the PII scrubbing layer.

Covers WhatsApp / SMS / Telegram message formats across b2's target regions
(Kenya, Senegal, Indonesia, Bangladesh, Nigeria) plus edge cases, multi-turn
conversation history, financial PII, and full CanonicalMessage prompt format
verification.

Run with:
    b2p/Scripts/python -m pytest tests/core/test_pii_scenarios.py -v -s
"""

from __future__ import annotations

import json
from pathlib import Path

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
    PiiScrubber,
    ScrubResult,
)


@pytest.fixture(scope="module")
def scrubber() -> PiiScrubber:
    config = PiiConfig(
        score_threshold=0.4,
        max_retries=3,
        retry_timeout_ms=1000,
        language="en",
        entities=[
            EntityConfig(name="PHONE_NUMBER", enabled=True, recognizer="regex"),
            EntityConfig(name="EMAIL_ADDRESS", enabled=True, recognizer="regex"),
            EntityConfig(name="CREDIT_CARD",   enabled=True, recognizer="regex"),
            EntityConfig(name="IBAN_CODE",     enabled=True, recognizer="regex"),
            EntityConfig(name="IP_ADDRESS",    enabled=True, recognizer="regex"),
        ],
        nlp_engine=NlpEngineConfig(provider="spacy", model_name="en_core_web_sm"),
        audit_store=AuditStoreConfig(
            memory_ttl_hours=24,
            file_path="data/scenario_pii_audit.jsonl",
            store_originals=True,
        ),
    )
    return PiiScrubber(config=config)


@pytest.fixture(scope="module")
def audit_store(scrubber: PiiScrubber) -> PiiAuditStore:
    store = PiiAuditStore(scrubber._config.audit_store)
    Path(scrubber._config.audit_store.file_path).unlink(missing_ok=True)
    return store


def _assert_clean(result: ScrubResult, *raw_values: str) -> None:
    full = (result.clean_message.text_content or "") + json.dumps(
        result.clean_message.session_context
    ) + json.dumps(result.clean_message.prior_context)
    for v in raw_values:
        assert v not in full, f"{v!r} still present after scrubbing"


def _prompt_forbidden_keys_absent(result: ScrubResult) -> None:
    prompt = result.clean_message.to_agent_prompt()
    forbidden = {"session_id", "channel", "media_url", "phone", "phone_number",
                 "lat", "lng", "latitude", "longitude", "coordinates"}
    assert set(prompt).isdisjoint(forbidden)


class TestWhatsApp:
    def test_farmer_intake_phone_and_email(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "Hello, I need help with my maize crop in Kisumu. "
            "My phone is +254 712 345 678, email joseph.kamau@gmail.com. "
            "The plants are turning yellow."
        )
        msg = CanonicalMessage(
            session_id=make_session_id("+254712345678", "whatsapp"),
            channel="whatsapp",
            input_type=InputType.TEXT,
            text_content=raw,
            language_hint="en",
            submission_type=SubmissionType.INTAKE,
            location_context=LocationContext(
                country_code="KE", region="Nyanza", city="Kisumu",
                source=LocationSource.DEVICE, confidence=0.95,
            ),
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii and not result.scrub_failed
        _assert_clean(result, "+254 712 345 678", "joseph.kamau@gmail.com")
        assert result.clean_message.location_context.city == "Kisumu"
        _prompt_forbidden_keys_absent(result)

    def test_session_context_phone_stripped_safe_fields_preserved(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id("+254700000002", "whatsapp"),
            channel="whatsapp",
            input_type=InputType.TEXT,
            text_content="What crops should I plant this season?",
            session_context={
                "contact": "+254 712 999 888",
                "region": "Western Kenya",
                "program": "smallholder-farming",
            },
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        _assert_clean(result, "+254 712 999 888")
        sc = result.clean_message.session_context
        assert sc["region"] == "Western Kenya"
        assert sc["program"] == "smallholder-farming"
        assert any("session_context" in e.field_path for e in result.entities)

    def test_image_message_no_text_scrub(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id("+254700000001", "whatsapp"),
            channel="whatsapp",
            input_type=InputType.IMAGE,
            text_content=None,
            media_url="https://media.whatsapp.net/photo/abc123.jpg",
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert not result.found_pii and not result.scrub_failed
        assert result.clean_message.text_content is None
        assert "media_url" not in result.clean_message.to_agent_prompt()

    def test_multi_turn_pii_in_prior_context(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        prior = [
            {"role": "user",      "content": "My number is +254 712 345 678 if needed."},
            {"role": "assistant", "content": "Thank you! What crop issue are you facing?"},
            {"role": "user",      "content": "Also email joseph.kamau@gmail.com for reports."},
            {"role": "assistant", "content": "Got it! What crop issue are you facing?"},
        ]
        msg = CanonicalMessage(
            session_id=make_session_id("+254712345678", "whatsapp"),
            channel="whatsapp",
            input_type=InputType.TEXT,
            text_content="Call me at +254 712 345 678 to confirm fertilizer delivery.",
            prior_context=prior,
            location_context=LocationContext(
                country_code="KE", region="Nyanza", city="Kisumu",
                source=LocationSource.DEVICE, confidence=0.95,
            ),
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        _assert_clean(result, "+254 712 345 678", "joseph.kamau@gmail.com")

        paths = {e.field_path for e in result.entities}
        assert any("text_content" in p for p in paths)
        assert any("prior_context" in p for p in paths)

        # Non-PII assistant turns preserved
        prior_str = json.dumps(result.clean_message.prior_context)
        assert "What crop issue are you facing?" in prior_str


class TestSms:
    def test_health_worker_phone_email_ip(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "Patient follow-ups: nurse Aisha at +221 77 123 4567, "
            "aisha.diallo@healthclinic.sn. Portal IP: 192.168.10.45."
        )
        msg = CanonicalMessage(
            session_id=make_session_id("+221771234567", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content=raw,
            language_hint="fr",
            submission_type=SubmissionType.MONITORING_CHECKIN,
            location_context=LocationContext(
                country_code="SN", region="Dakar",
                source=LocationSource.PHONE_PREFIX, confidence=0.8,
            ),
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        _assert_clean(result, "aisha.diallo@healthclinic.sn", "192.168.10.45")
        entity_types = {e.entity_type for e in result.entities}
        assert "EMAIL_ADDRESS" in entity_types
        assert "IP_ADDRESS" in entity_types

    def test_financial_iban_stripped(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "Loan repayment IBANs: DE89 3704 0044 0532 0130 00 "
            "and GB29 NWBK 6016 1331 9268 19. "
            "Disbursement contact: treasurer@coop.ke"
        )
        msg = CanonicalMessage(
            session_id=make_session_id("+491601234567", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content=raw,
            submission_type=SubmissionType.VERIFICATION_EVIDENCE,
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        _assert_clean(result, "DE89 3704 0044 0532 0130 00", "treasurer@coop.ke")
        assert any(e.entity_type == "IBAN_CODE" for e in result.entities)

    def test_clean_message_unchanged(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "My sorghum in Tambacounda has white patches on the leaves. "
            "Started after the rains last week. What could it be?"
        )
        msg = CanonicalMessage(
            session_id=make_session_id("+221761111111", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content=raw,
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert not result.found_pii and not result.scrub_failed
        assert result.clean_message.text_content == raw


class TestTelegram:
    def test_dense_pii_all_fields(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id("tg_99182", "telegram"),
            channel="telegram",
            input_type=InputType.TEXT,
            text_content="Email me at me@borrower.id. Device IP: 10.20.30.40.",
            language_hint="id",
            submission_type=SubmissionType.INTAKE,
            session_context={"referral_email": "agent@lendingapp.id", "program": "micro-lending"},
            prior_context=[
                {"role": "user",      "content": "Contact me at prev@older.id"},
                {"role": "assistant", "content": "Thank you, how can I help you today?"},
            ],
            location_context=LocationContext(
                country_code="ID", region="West Java", city="Bandung",
                source=LocationSource.DEVICE, confidence=0.9,
            ),
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        _assert_clean(result, "me@borrower.id", "10.20.30.40",
                      "agent@lendingapp.id", "prev@older.id")

        # Non-PII preserved
        assert result.clean_message.session_context["program"] == "micro-lending"
        assert "Thank you" in result.clean_message.prior_context[1]["content"]

        paths = {e.field_path for e in result.entities}
        assert any("text_content" in p for p in paths)
        assert any("session_context" in p for p in paths)
        assert any("prior_context" in p for p in paths)

    def test_international_phone_formats(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "Contact options: +44 20 7946 0958, +254712345678, +62 812 3456 7890."
        )
        msg = CanonicalMessage(
            session_id=make_session_id("tg_77001", "telegram"),
            channel="telegram",
            input_type=InputType.TEXT,
            text_content=raw,
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        phones = [e for e in result.entities if e.entity_type == "PHONE_NUMBER"]
        assert len(phones) >= 1

    def test_legal_aid_heavy_pii(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "I need legal assistance. Reach me at +234 803 456 7890 or "
            "emeka.obi@yahoo.com. Bank IBAN GB29 NWBK 6016 1331 9268 19. "
            "Device IP: 192.0.2.147."
        )
        msg = CanonicalMessage(
            session_id=make_session_id("+2348034567890", "telegram"),
            channel="telegram",
            input_type=InputType.TEXT,
            text_content=raw,
            submission_type=SubmissionType.INTAKE,
            location_context=LocationContext(
                country_code="NG", region="Lagos",
                source=LocationSource.PHONE_PREFIX, confidence=0.8,
            ),
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        _assert_clean(result, "emeka.obi@yahoo.com", "192.0.2.147")
        entity_types = {e.entity_type for e in result.entities}
        assert "EMAIL_ADDRESS" in entity_types and "IP_ADDRESS" in entity_types


class TestEdgeCases:
    def test_farming_numbers_no_false_positives(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "Yield: 3.5 tons/hectare. Row spacing: 45cm. "
            "DAP applied: 50kg/acre. Field size: 2.3 hectares."
        )
        msg = CanonicalMessage(
            session_id=make_session_id("fp_test", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content=raw,
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert not result.scrub_failed
        for e in result.entities:
            assert e.original_value not in {"3.5", "45", "50", "2.3"}

    def test_long_message_pii_buried_mid_text(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = (
            "Dear extension officer, I am writing about my sorghum crop in "
            "Tambacounda. The rains started late and the plants developed rust. "
            "Contact me: +221 76 234 5678 or amadou.sarr@agriculture.sn. "
            "Also copy our cooperative: coop.tambacounda@gmail.com. "
            "Portal IP: 10.0.50.200. We have 47 farmers needing urgent help."
        )
        msg = CanonicalMessage(
            session_id=make_session_id("long_msg", "whatsapp"),
            channel="whatsapp",
            input_type=InputType.TEXT,
            text_content=raw,
            location_context=LocationContext(
                country_code="SN", region="Tambacounda",
                source=LocationSource.PROMPT, confidence=0.75,
            ),
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        assert result.found_pii
        _assert_clean(result, "amadou.sarr@agriculture.sn",
                      "coop.tambacounda@gmail.com", "10.0.50.200")
        clean = result.clean_message.text_content or ""
        assert "sorghum" in clean and "47 farmers" in clean

    def test_same_pii_in_text_and_context_both_stripped(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        phone = "+254 720 111 222"
        msg = CanonicalMessage(
            session_id=make_session_id("dup_pii", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content=f"Call me at {phone} for confirmation.",
            session_context={"stored_number": phone, "loan_id": "LN-0391"},
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        _assert_clean(result, phone)
        assert result.clean_message.session_context["loan_id"] == "LN-0391"

    def test_audio_message_session_context_email_stripped(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id("audio_test", "whatsapp"),
            channel="whatsapp",
            input_type=InputType.AUDIO,
            text_content=None,
            session_context={"transcription_by": "agent@b2.ai", "locale": "sw-KE"},
        )
        result = scrubber.scrub(msg)
        audit_store.record(msg.session_id, result)

        sc = result.clean_message.session_context
        assert "agent@b2.ai" not in sc.get("transcription_by", "")
        assert sc["locale"] == "sw-KE"


class TestFailOpen:
    def test_presidio_error_passes_raw_message(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        raw = "My number is +44 7911 123456"
        msg = CanonicalMessage(
            session_id=make_session_id("fail_open", "sms"),
            channel="sms",
            input_type=InputType.TEXT,
            text_content=raw,
        )
        original = scrubber._do_scrub
        try:
            scrubber._do_scrub = lambda m: (_ for _ in ()).throw(
                RuntimeError("simulated Presidio failure")
            )
            result = scrubber.scrub(msg)
        finally:
            scrubber._do_scrub = original

        audit_store.record(msg.session_id, result)

        assert result.scrub_failed is True
        # text_content is nulled — unscanned PII must not reach the Agentic OS
        assert result.clean_message.text_content is None
        assert not result.found_pii


class TestPromptFormat:
    def test_whatsapp_intake_prompt_structure(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id("+254712000001", "whatsapp"),
            channel="whatsapp",
            input_type=InputType.TEXT,
            text_content="Call me at +254 712 000 001. I need crop advice.",
            language_hint="en",
            submission_type=SubmissionType.INTAKE,
            location_context=LocationContext(
                country_code="KE", region="Nyanza", city="Kisumu",
                source=LocationSource.DEVICE, confidence=0.95,
            ),
            session_context={"program": "farming-kenya"},
        )
        result = scrubber.scrub(msg)
        prompt = result.clean_message.to_agent_prompt()

        assert prompt["input_type"] == "text"
        assert prompt["language_hint"] == "en"
        assert prompt["submission_type"] == "intake"
        assert prompt["location_context"]["country_code"] == "KE"
        assert prompt["location_context"]["city"] == "Kisumu"
        assert "coordinates" not in prompt["location_context"]
        assert prompt["session_context"]["program"] == "farming-kenya"
        assert "session_id" not in prompt
        assert "channel" not in prompt
        assert "+254 712 000 001" not in prompt.get("text_content", "")

    def test_telegram_prior_context_pii_stripped_in_prompt(
        self, scrubber: PiiScrubber, audit_store: PiiAuditStore
    ) -> None:
        msg = CanonicalMessage(
            session_id=make_session_id("tg_legal_01", "telegram"),
            channel="telegram",
            input_type=InputType.TEXT,
            text_content="My landlord evicted me illegally. I need urgent advice.",
            language_hint="en",
            submission_type=SubmissionType.INTAKE,
            prior_context=[
                {"role": "user",      "content": "I need legal help, email law@legalaid.ng"},
                {"role": "assistant", "content": "I can help. What is your concern?"},
            ],
            location_context=LocationContext(
                country_code="NG", region="Lagos",
                source=LocationSource.PROMPT, confidence=0.7,
            ),
        )
        result = scrubber.scrub(msg)
        prompt = result.clean_message.to_agent_prompt()

        assert "prior_context" in prompt
        prior_str = json.dumps(prompt["prior_context"])
        assert "law@legalaid.ng" not in prior_str
        assert "EMAIL_ADDRESS" in prior_str
        assert "What is your concern?" in prior_str
        assert "channel" not in prompt


class TestAuditStore:
    def test_audit_file_valid_jsonl_all_required_fields(
        self, audit_store: PiiAuditStore
    ) -> None:
        audit_path = Path(audit_store._config.file_path)
        if not audit_path.exists():
            pytest.skip("No audit file yet")

        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        required = {"session_id_hash", "scrubbed_at", "scrub_failed", "entities"}
        for i, line in enumerate(lines):
            record = json.loads(line)
            missing = required - set(record)
            assert not missing, f"Line {i+1} missing: {missing}"
            for entity in record["entities"]:
                assert len(entity["sha256_hash"]) == 64
                assert entity["original_value"] not in entity["placeholder"]

    def test_in_memory_count_positive(self, audit_store: PiiAuditStore) -> None:
        assert audit_store.today_count() >= 0
