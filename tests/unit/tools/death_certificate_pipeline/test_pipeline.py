"""POC-1 unit tests.

Covers every acceptance criterion from the GitHub issue plus edge cases:

1. All models are typed and constructable
2. narrative is a first-class field with blank validation
3. run_pipeline() runs end-to-end on stubs and returns valid ReliabilityResult
4. band enum covers high, medium, low, escalate
5. score is int in [1, 100]
6. sub_scores has the three expected keys; weights sum to 1.0
7. Hard escalation from authenticity stage forces Band.ESCALATE
8. FastAPI POST /score accepts image + narrative + fields and returns JSON
9. CLI argument parser is wired correctly (smoke test)

POC-2 alignment checks:
- AuthenticitySignal correctly wraps ToolResult from fake_image_detector
- ConsistencySignal field names match the shape analyze_death_certificate_consistency returns
- extracted_fields passes through from ConsistencySignal to ReliabilityResult
"""

from __future__ import annotations

import io
import json

import pytest

from poc.models import (
    AuthenticitySignal,
    Band,
    ConsistencySignal,
    DocumentSignal,
    ReliabilityResult,
    Submission,
)
from poc.pipeline import _stage_score, run_pipeline
from tools.fake_image_detector.models import Escalation, ToolResult, Verdict

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FAKE_IMAGE = b"\x89PNG\r\n\x1a\n"  # minimal PNG magic bytes
NARRATIVE  = "My father John Smith passed away on March 3rd, 2021 in Chicago."


@pytest.fixture()
def minimal_submission() -> Submission:
    return Submission(image=FAKE_IMAGE, narrative=NARRATIVE)


# ---------------------------------------------------------------------------
# 1. Model construction
# ---------------------------------------------------------------------------

class TestModelConstruction:
    def test_submission_constructs_with_required_fields(self) -> None:
        s = Submission(image=FAKE_IMAGE, narrative=NARRATIVE)
        assert s.image == FAKE_IMAGE
        assert s.narrative == NARRATIVE
        assert s.case_fields == {}

    def test_submission_accepts_case_fields(self) -> None:
        s = Submission(image=FAKE_IMAGE, narrative=NARRATIVE, case_fields={"case_id": "ABC-1"})
        assert s.case_fields["case_id"] == "ABC-1"

    def test_document_signal_defaults(self) -> None:
        sig = DocumentSignal()
        assert sig.legible is False
        assert sig.document_type is None
        assert sig.page_count == 1
        assert sig.stage == "document_analysis"

    def test_authenticity_signal_wraps_tool_result(self) -> None:
        tr = ToolResult(verdict=Verdict.PASS, risk_score=0.1,
                        escalation=Escalation.AUTO_ACCEPT, checks=[])
        sig = AuthenticitySignal(result=tr)
        assert sig.result.verdict == Verdict.PASS
        assert sig.stage == "authenticity"

    def test_consistency_signal_defaults(self) -> None:
        sig = ConsistencySignal()
        assert sig.consistency_score == 0.0
        assert sig.extracted_fields == {}
        assert sig.stage == "consistency"

    def test_reliability_result_constructs(self) -> None:
        r = ReliabilityResult(
            score=88,
            band=Band.HIGH,
            sub_scores={"document": 1.0, "authenticity": 0.9, "consistency": 0.8},
            weights={"document": 0.2, "authenticity": 0.4, "consistency": 0.4},
            flags=[],
            justification="test",
        )
        assert r.score == 88
        assert r.band == Band.HIGH


# ---------------------------------------------------------------------------
# 2. Submission.narrative validation
# ---------------------------------------------------------------------------

class TestNarrativeValidation:
    def test_blank_narrative_raises(self) -> None:
        with pytest.raises(ValueError, match="narrative must not be blank"):
            Submission(image=FAKE_IMAGE, narrative="")

    def test_whitespace_only_narrative_raises(self) -> None:
        with pytest.raises(ValueError, match="narrative must not be blank"):
            Submission(image=FAKE_IMAGE, narrative="   \t\n  ")

    def test_valid_narrative_is_accepted(self) -> None:
        s = Submission(image=FAKE_IMAGE, narrative="valid")
        assert s.narrative == "valid"


# ---------------------------------------------------------------------------
# 3. Band enum completeness
# ---------------------------------------------------------------------------

class TestBandEnum:
    def test_band_covers_all_four_values(self) -> None:
        values = {b.value for b in Band}
        assert values == {"high", "medium", "low", "escalate"}

    def test_band_is_string_enum(self) -> None:
        assert Band.HIGH == "high"
        assert Band.ESCALATE == "escalate"


# ---------------------------------------------------------------------------
# 4. run_pipeline end-to-end on stubs
# ---------------------------------------------------------------------------

class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_returns_reliability_result(
        self, minimal_submission: Submission
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert isinstance(result, ReliabilityResult)

    @pytest.mark.asyncio
    async def test_score_is_int_in_range(
        self, minimal_submission: Submission
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert isinstance(result.score, int)
        assert 1 <= result.score <= 100

    @pytest.mark.asyncio
    async def test_band_is_valid_enum_value(
        self, minimal_submission: Submission
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert result.band in list(Band)

    @pytest.mark.asyncio
    async def test_sub_scores_has_three_stage_keys(
        self, minimal_submission: Submission
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert set(result.sub_scores) == {"document", "authenticity", "consistency"}

    @pytest.mark.asyncio
    async def test_weights_sum_to_one(
        self, minimal_submission: Submission
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert abs(sum(result.weights.values()) - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_justification_is_non_empty_string(
        self, minimal_submission: Submission
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert isinstance(result.justification, str)
        assert len(result.justification) > 0

    @pytest.mark.asyncio
    async def test_extracted_fields_is_dict(
        self, minimal_submission: Submission
    ) -> None:
        result = await run_pipeline(minimal_submission)
        assert isinstance(result.extracted_fields, dict)

    @pytest.mark.asyncio
    async def test_default_stubs_produce_high_band(
        self, minimal_submission: Submission
    ) -> None:
        # Stub defaults: doc=1.0, auth=0.9, con=0.8
        # weighted = 1.0×0.2 + 0.9×0.4 + 0.8×0.4 = 0.88 → score=88 → HIGH
        result = await run_pipeline(minimal_submission)
        assert result.score == 88
        assert result.band == Band.HIGH


# ---------------------------------------------------------------------------
# 5. Scorer — hard escalation overrides band
# ---------------------------------------------------------------------------

class TestScorerEscalation:
    @pytest.mark.asyncio
    async def test_human_review_forces_escalate_band(self) -> None:
        doc  = DocumentSignal(legible=True, document_type="death_certificate")
        auth = AuthenticitySignal(
            result=ToolResult(
                verdict=Verdict.FLAG,
                risk_score=0.9,
                escalation=Escalation.HUMAN_REVIEW,
                checks=[],
            )
        )
        con  = ConsistencySignal(consistency_score=0.8, consistency_label="high")

        result = await _stage_score(doc, auth, con)

        assert result.band == Band.ESCALATE
        assert "HARD_ESCALATION" in result.flags

    @pytest.mark.asyncio
    async def test_auto_reject_forces_escalate_band(self) -> None:
        doc  = DocumentSignal(legible=True)
        auth = AuthenticitySignal(
            result=ToolResult(
                verdict=Verdict.REJECT,
                risk_score=1.0,
                escalation=Escalation.AUTO_REJECT,
                checks=[],
            )
        )
        con  = ConsistencySignal(consistency_score=0.9, consistency_label="high")

        result = await _stage_score(doc, auth, con)

        assert result.band == Band.ESCALATE
        assert "HARD_ESCALATION" in result.flags

    @pytest.mark.asyncio
    async def test_low_scores_produce_low_band(self) -> None:
        doc  = DocumentSignal(legible=False)            # doc_score = 0.3
        auth = AuthenticitySignal(
            result=ToolResult(
                verdict=Verdict.FLAG,
                risk_score=0.6,                         # auth_score = 0.4
                escalation=Escalation.AUTO_ACCEPT,
                checks=[],
            )
        )
        con  = ConsistencySignal(consistency_score=0.2) # con_score = 0.2

        result = await _stage_score(doc, auth, con)

        # weighted = 0.3×0.2 + 0.4×0.4 + 0.2×0.4 = 0.06+0.16+0.08 = 0.30
        # score = 30 → LOW
        assert result.band == Band.LOW
        assert result.score == 30

    @pytest.mark.asyncio
    async def test_very_low_scores_produce_escalate_band(self) -> None:
        doc  = DocumentSignal(legible=False)            # doc_score = 0.3
        auth = AuthenticitySignal(
            result=ToolResult(
                verdict=Verdict.FLAG,
                risk_score=0.9,                         # auth_score = 0.1
                escalation=Escalation.AUTO_ACCEPT,
                checks=[],
            )
        )
        con  = ConsistencySignal(consistency_score=0.0) # con_score = 0.0

        result = await _stage_score(doc, auth, con)

        # weighted = 0.3×0.2 + 0.1×0.4 + 0.0×0.4 = 0.06+0.04+0.0 = 0.10
        # score = 10 → ESCALATE (< 25)
        assert result.band == Band.ESCALATE
        assert result.score == 10


# ---------------------------------------------------------------------------
# 6. Scorer — extracted_fields passthrough
# ---------------------------------------------------------------------------

class TestExtractedFieldsPassthrough:
    @pytest.mark.asyncio
    async def test_extracted_fields_passed_to_result(self) -> None:
        """POC-2 alignment: certificate data from ConsistencySignal reaches
        ReliabilityResult.extracted_fields unchanged."""
        doc  = DocumentSignal(legible=True)
        auth = AuthenticitySignal(result=ToolResult(
            verdict=Verdict.PASS, risk_score=0.1,
            escalation=Escalation.AUTO_ACCEPT, checks=[],
        ))
        con  = ConsistencySignal(
            consistency_score=0.9,
            extracted_fields={
                "full_name": "Jane Doe",
                "date_of_death": "2024-05-01",
                "cause_of_death": "cardiac arrest",
            },
        )

        result = await _stage_score(doc, auth, con)

        assert result.extracted_fields["full_name"] == "Jane Doe"
        assert result.extracted_fields["date_of_death"] == "2024-05-01"
        assert result.extracted_fields["cause_of_death"] == "cardiac arrest"


# ---------------------------------------------------------------------------
# 7. FastAPI endpoint
# ---------------------------------------------------------------------------

class TestApiEndpoint:
    @pytest.mark.asyncio
    async def test_score_endpoint_returns_200_and_valid_json(self) -> None:
        import httpx
        from poc.api import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": NARRATIVE, "case_fields": "{}"},
                files={"image": ("cert.png", io.BytesIO(FAKE_IMAGE), "image/png")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "band" in data
        assert "sub_scores" in data
        assert "weights" in data
        assert "justification" in data
        assert "extracted_fields" in data

    @pytest.mark.asyncio
    async def test_score_endpoint_rejects_blank_narrative(self) -> None:
        import httpx
        from poc.api import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": "   ", "case_fields": "{}"},
                files={"image": ("cert.png", io.BytesIO(FAKE_IMAGE), "image/png")},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_score_endpoint_rejects_invalid_case_fields_json(self) -> None:
        import httpx
        from poc.api import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": NARRATIVE, "case_fields": "not-json"},
                files={"image": ("cert.png", io.BytesIO(FAKE_IMAGE), "image/png")},
            )

        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_score_endpoint_band_is_known_value(self) -> None:
        import httpx
        from poc.api import app

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/score",
                data={"narrative": NARRATIVE, "case_fields": "{}"},
                files={"image": ("cert.png", io.BytesIO(FAKE_IMAGE), "image/png")},
            )

        band = response.json()["band"]
        assert band in {"high", "medium", "low", "escalate"}


# ---------------------------------------------------------------------------
# 8. CLI — argument parser and image resolution
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_parser_builds_without_error(self) -> None:
        from poc.cli import _build_parser
        parser = _build_parser()
        assert parser is not None

    def test_cli_score_subcommand_parses_args(self, tmp_path) -> None:
        from poc.cli import _build_parser

        image_file = tmp_path / "cert.png"
        image_file.write_bytes(FAKE_IMAGE)

        parser = _build_parser()
        args = parser.parse_args([
            "score",
            "--image", str(image_file),
            "--narrative", NARRATIVE,
            "--fields", '{"case_id": "X1"}',
        ])

        assert args.command == "score"
        assert args.narrative == NARRATIVE
        assert json.loads(args.fields) == {"case_id": "X1"}

    @pytest.mark.asyncio
    async def test_resolve_image_reads_local_file(self, tmp_path) -> None:
        from poc.cli import _resolve_image

        image_file = tmp_path / "cert.png"
        image_file.write_bytes(FAKE_IMAGE)

        result = await _resolve_image(str(image_file))
        assert result == FAKE_IMAGE

    @pytest.mark.asyncio
    async def test_resolve_image_local_missing_file_exits(self, tmp_path) -> None:
        from poc.cli import _resolve_image

        with pytest.raises(SystemExit):
            await _resolve_image(str(tmp_path / "nonexistent.png"))

    @pytest.mark.asyncio
    async def test_resolve_image_downloads_from_gcs(self, monkeypatch) -> None:
        """GCS URI is parsed correctly and download_as_bytes is called."""
        from unittest.mock import MagicMock
        from poc.cli import _resolve_image

        mock_blob = MagicMock()
        mock_blob.download_as_bytes.return_value = FAKE_IMAGE

        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        mock_storage = MagicMock()
        mock_storage.Client.return_value = mock_client

        monkeypatch.setitem(__import__("sys").modules, "google.cloud.storage", mock_storage)
        monkeypatch.setitem(__import__("sys").modules, "google.cloud", MagicMock(storage=mock_storage))

        result = await _resolve_image("gs://my-bucket/certs/cert.jpg")

        mock_client.bucket.assert_called_once_with("my-bucket")
        mock_bucket.blob.assert_called_once_with("certs/cert.jpg")
        mock_blob.download_as_bytes.assert_called_once()
        assert result == FAKE_IMAGE

    @pytest.mark.asyncio
    async def test_resolve_image_invalid_gcs_uri_exits(self, monkeypatch) -> None:
        """A GCS URI with no blob path exits with an error."""
        from unittest.mock import MagicMock
        from poc.cli import _resolve_image

        mock_storage = MagicMock()
        monkeypatch.setitem(__import__("sys").modules, "google.cloud.storage", mock_storage)
        monkeypatch.setitem(__import__("sys").modules, "google.cloud", MagicMock(storage=mock_storage))

        with pytest.raises(SystemExit):
            await _resolve_image("gs://bucket-only-no-blob")
