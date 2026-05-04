import json
import shutil
import tempfile
import uuid
from io import BytesIO
from unittest.mock import MagicMock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from rest_framework.test import APIClient

from .models import Document

TEST_MEDIA_ROOT = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(TEST_MEDIA_ROOT, ignore_errors=True)


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class DocumentUploadTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("qa.views.ingest_document", return_value=10)
    def test_upload_pdf(self, mock_ingest):
        pdf_content = b"%PDF-1.4 fake pdf content"
        file = SimpleUploadedFile("test.pdf", pdf_content, content_type="application/pdf")

        response = self.client.post("/api/documents/", {"file": file}, format="multipart")

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["filename"], "test.pdf")
        self.assertEqual(data["file_type"], "pdf")
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["chunk_count"], 10)
        mock_ingest.assert_called_once()

    @patch("qa.views.ingest_document", return_value=5)
    def test_upload_json(self, mock_ingest):
        json_content = json.dumps([{"key": "value"}]).encode()
        file = SimpleUploadedFile("data.json", json_content, content_type="application/json")

        response = self.client.post("/api/documents/", {"file": file}, format="multipart")

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["file_type"], "json")
        self.assertEqual(data["status"], "ready")

    def test_upload_invalid_type(self):
        file = SimpleUploadedFile("data.txt", b"text content", content_type="text/plain")

        response = self.client.post("/api/documents/", {"file": file}, format="multipart")

        self.assertEqual(response.status_code, 400)

    def test_upload_no_file(self):
        response = self.client.post("/api/documents/", {}, format="multipart")

        self.assertEqual(response.status_code, 400)

    @patch("qa.views.ingest_document", side_effect=Exception("Ingest failed"))
    def test_upload_ingest_failure(self, mock_ingest):
        file = SimpleUploadedFile("test.pdf", b"%PDF-1.4 content", content_type="application/pdf")

        response = self.client.post("/api/documents/", {"file": file}, format="multipart")

        self.assertEqual(response.status_code, 500)
        doc = Document.objects.first()
        self.assertEqual(doc.status, "failed")


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class DocumentDetailTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.document = Document.objects.create(
            filename="test.pdf",
            file_type="pdf",
            file=SimpleUploadedFile("test.pdf", b"%PDF"),
            status="ready",
            chunk_count=42,
        )

    def test_get_document(self):
        response = self.client.get(f"/api/documents/{self.document.id}/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "test.pdf")
        self.assertEqual(data["status"], "ready")
        self.assertEqual(data["chunk_count"], 42)

    def test_get_nonexistent_document(self):
        fake_id = uuid.uuid4()
        response = self.client.get(f"/api/documents/{fake_id}/")

        self.assertEqual(response.status_code, 404)


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class QuestionsTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.document = Document.objects.create(
            filename="test.pdf",
            file_type="pdf",
            file=SimpleUploadedFile("test.pdf", b"%PDF"),
            status="ready",
            chunk_count=10,
        )

    @patch("qa.views.answer_questions")
    def test_ask_questions_json(self, mock_answer):
        mock_answer.return_value = [
            {
                "question": "What is this?",
                "answer": "A test document.",
                "confidence": "high",
                "citations": [{"source": "test.pdf", "page": 0, "excerpt": "..."}],
                "error": None,
            }
        ]

        response = self.client.post(
            f"/api/documents/{self.document.id}/questions/",
            {"questions": ["What is this?"]},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["answer"], "A test document.")
        self.assertEqual(data[0]["confidence"], "high")
        self.assertIsNotNone(data[0]["citations"])

    @patch("qa.views.answer_questions")
    def test_ask_questions_file(self, mock_answer):
        mock_answer.return_value = [
            {"question": "Q1?", "answer": "A1", "confidence": "medium", "citations": [], "error": None}
        ]
        questions_file = SimpleUploadedFile(
            "questions.json",
            json.dumps(["Q1?"]).encode(),
            content_type="application/json",
        )

        response = self.client.post(
            f"/api/documents/{self.document.id}/questions/",
            {"questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 200)

    def test_ask_questions_not_ready(self):
        self.document.status = "processing"
        self.document.save()

        response = self.client.post(
            f"/api/documents/{self.document.id}/questions/",
            {"questions": ["What?"]},
            format="json",
        )

        self.assertEqual(response.status_code, 400)

    def test_ask_questions_no_questions(self):
        response = self.client.post(
            f"/api/documents/{self.document.id}/questions/",
            {},
            format="json",
        )

        self.assertEqual(response.status_code, 400)

    def test_ask_questions_nonexistent_doc(self):
        fake_id = uuid.uuid4()
        response = self.client.post(
            f"/api/documents/{fake_id}/questions/",
            {"questions": ["What?"]},
            format="json",
        )

        self.assertEqual(response.status_code, 404)

    @patch("qa.views.answer_questions")
    def test_partial_failure(self, mock_answer):
        mock_answer.return_value = [
            {"question": "Q1?", "answer": "A1", "confidence": "high", "citations": [], "error": None},
            {"question": "Q2?", "answer": None, "confidence": None, "citations": [],
             "error": {"code": "llm_timeout", "message": "The model call timed out."}},
        ]

        response = self.client.post(
            f"/api/documents/{self.document.id}/questions/",
            {"questions": ["Q1?", "Q2?"]},
            format="json",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertIsNone(data[1]["answer"])
        self.assertEqual(data[1]["error"]["code"], "llm_timeout")


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class ServiceUnitTests(TestCase):
    def test_multiline_option_question_expands_retrieval_queries(self):
        from .services import _expand_retrieval_queries

        queries = _expand_retrieval_queries(
            "Which of the following are performed as part of monitoring:\n"
            "Application Performance Monitoring (APM)\n"
            "End User Monitoring (EUM)\n"
            "Digital Experience Monitoring (DEM)"
        )

        self.assertIn("Which of the following are performed as part of monitoring:", queries)
        self.assertIn("Application Performance Monitoring (APM)", queries)
        self.assertIn("Application Performance Monitoring", queries)
        self.assertIn("APM", queries)
        self.assertIn("End User Monitoring", queries)
        self.assertIn("Digital Experience Monitoring", queries)
        self.assertIn("DEM", queries)

    @patch("qa.services._get_llm")
    @patch("qa.services._get_vector_store")
    def test_answer_single_question(self, mock_vs, mock_llm):
        from langchain_core.documents import Document as LCDocument

        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (
                LCDocument(
                    page_content="AWS is used as the cloud provider.",
                    metadata={"source": "test.pdf", "page": 5, "chunk_index": 0},
                ),
                0.91,
            )
        ]
        mock_vs.return_value = mock_store

        mock_response = MagicMock()
        mock_response.content = json.dumps({"answer": "AWS", "confidence": "high"})
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 10}}
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from .services import _answer_single_question

        result = _answer_single_question("Which cloud providers?", document_id="test-doc-id")

        self.assertEqual(result.answer, "AWS")
        self.assertEqual(result.confidence, "high")
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0]["page"], 6)

    @patch("qa.services._get_vector_store")
    def test_answer_no_results(self, mock_vs):
        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = []
        mock_vs.return_value = mock_store

        from .services import _answer_single_question

        result = _answer_single_question("Unknown question?")

        self.assertIn("could not find", result.answer.lower())
        self.assertEqual(result.confidence, "low")

    @patch("qa.services._get_llm")
    @patch("qa.services._get_vector_store")
    def test_answer_low_relevance_falls_back_to_top_result(self, mock_vs, mock_llm):
        from langchain_core.documents import Document as LCDocument

        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (
                LCDocument(
                    page_content="Unrelated content.",
                    metadata={"source": "test.pdf", "page": 0, "chunk_index": 0},
                ),
                0.01,
            )
        ]
        mock_vs.return_value = mock_store
        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "answer": "I could not find this information in the provided documents.",
            "confidence": "low",
        })
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 50, "completion_tokens": 10}}
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from .services import _answer_single_question

        result = _answer_single_question("Which cloud providers?")

        self.assertIn("could not find", result.answer.lower())
        self.assertEqual(result.confidence, "low")
        self.assertEqual(len(result.citations), 1)
        self.assertTrue(result.citations[0]["relevance_score"] < 0.2)
        mock_llm.assert_called_once()

    @patch("qa.services._get_llm")
    @patch("qa.services._get_vector_store")
    def test_not_found_answer_is_low_confidence(self, mock_vs, mock_llm):
        from langchain_core.documents import Document as LCDocument

        mock_store = MagicMock()
        mock_store.similarity_search_with_relevance_scores.return_value = [
            (
                LCDocument(
                    page_content="Monitoring is described in general terms.",
                    metadata={"source": "test.pdf", "page": 3, "chunk_index": 1},
                ),
                0.8,
            )
        ]
        mock_vs.return_value = mock_store

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "answer": "I could not find this information in the provided documents.",
            "confidence": "medium",
        })
        mock_response.response_metadata = {"token_usage": {"prompt_tokens": 50, "completion_tokens": 10}}
        mock_llm_instance = MagicMock()
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance

        from .services import _answer_single_question

        result = _answer_single_question("Which monitoring processes are used?")

        self.assertIn("could not find", result.answer.lower())
        self.assertEqual(result.confidence, "low")

    @patch("qa.services.ThreadPoolExecutor")
    def test_answer_questions_timeout_preserves_question(self, mock_executor_cls):
        from concurrent.futures import TimeoutError as FuturesTimeoutError

        mock_future = MagicMock()
        mock_future.result.side_effect = FuturesTimeoutError()
        mock_executor = MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.submit.return_value = mock_future
        mock_executor_cls.return_value = mock_executor

        from .services import answer_questions

        results = answer_questions(["Original question?"])

        self.assertEqual(results[0]["question"], "Original question?")
        self.assertEqual(results[0]["error"]["code"], "llm_timeout")


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class AllDocumentsQuestionsTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("qa.views.answer_questions")
    def test_ask_all_documents(self, mock_answer):
        Document.objects.create(
            filename="a.pdf", file_type="pdf",
            file=SimpleUploadedFile("a.pdf", b"%PDF"), status="ready", chunk_count=5,
        )
        mock_answer.return_value = [
            {"question": "Q?", "answer": "A", "confidence": "high",
             "citations": [{"document_id": "x", "source": "a.pdf", "excerpt": "..."}],
             "error": None}
        ]

        response = self.client.post(
            "/api/questions/", {"questions": ["Q?"]}, format="json",
        )

        self.assertEqual(response.status_code, 200)
        mock_answer.assert_called_once_with(["Q?"])

    def test_ask_all_no_documents(self):
        response = self.client.post(
            "/api/questions/", {"questions": ["Q?"]}, format="json",
        )

        self.assertEqual(response.status_code, 400)


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class OneShotAnswerExtendedTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("qa.views.answer_questions")
    @patch("qa.views.ingest_document", return_value=7)
    def test_one_shot_pdf_with_questions_file(self, mock_ingest, mock_answer):
        mock_answer.return_value = [
            {
                "question": "Which cloud providers?",
                "answer": "AWS",
                "confidence": "high",
                "citations": [{"source": "soc2.pdf", "page": 1, "chunk_index": 0, "excerpt": "AWS"}],
                "error": None,
            }
        ]
        document_file = SimpleUploadedFile("soc2.pdf", b"%PDF-1.4 content", content_type="application/pdf")
        questions_file = SimpleUploadedFile(
            "questions.json",
            json.dumps(["Which cloud providers?"]).encode(),
            content_type="application/json",
        )

        response = self.client.post(
            "/api/answer/",
            {"document_file": document_file, "questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["document"]["filename"], "soc2.pdf")
        self.assertEqual(data["document"]["status"], "ready")
        self.assertEqual(data["document"]["chunk_count"], 7)
        self.assertEqual(data["results"][0]["answer"], "AWS")
        mock_ingest.assert_called_once()
        mock_answer.assert_called_once()

    @patch("qa.views.answer_questions")
    @patch("qa.views.ingest_document", return_value=3)
    def test_one_shot_json_document_with_object_questions_file(self, mock_ingest, mock_answer):
        mock_answer.return_value = [
            {"question": "Q?", "answer": "A", "confidence": "medium", "citations": [], "error": None}
        ]
        document_file = SimpleUploadedFile("data.json", b'{"vendor": "zania"}', content_type="application/json")
        questions_file = SimpleUploadedFile(
            "questions.json",
            json.dumps({"questions": ["Q?"]}).encode(),
            content_type="application/json",
        )

        response = self.client.post(
            "/api/answer/",
            {"document_file": document_file, "questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document"]["file_type"], "json")
        mock_answer.assert_called_once()

    def test_one_shot_invalid_document_extension(self):
        document_file = SimpleUploadedFile("notes.txt", b"content", content_type="text/plain")
        questions_file = SimpleUploadedFile("questions.json", json.dumps(["Q?"]).encode())

        response = self.client.post(
            "/api/answer/",
            {"document_file": document_file, "questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 400)

    def test_one_shot_invalid_questions_json(self):
        document_file = SimpleUploadedFile("data.json", b'{"ok": true}', content_type="application/json")
        questions_file = SimpleUploadedFile("questions.json", b"{invalid", content_type="application/json")

        response = self.client.post(
            "/api/answer/",
            {"document_file": document_file, "questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 400)

    def test_one_shot_too_many_questions(self):
        document_file = SimpleUploadedFile("data.json", b'{"ok": true}', content_type="application/json")
        questions_file = SimpleUploadedFile(
            "questions.json",
            json.dumps(["Q?"] * 21).encode(),
            content_type="application/json",
        )

        response = self.client.post(
            "/api/answer/",
            {"document_file": document_file, "questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 400)

    @patch("qa.views.ingest_document", side_effect=Exception("bad pdf"))
    def test_one_shot_ingest_failure_marks_failed(self, mock_ingest):
        document_file = SimpleUploadedFile("bad.pdf", b"%PDF-1.4 bad", content_type="application/pdf")
        questions_file = SimpleUploadedFile("questions.json", json.dumps(["Q?"]).encode())

        response = self.client.post(
            "/api/answer/",
            {"document_file": document_file, "questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 500)
        document = Document.objects.first()
        self.assertEqual(document.status, "failed")
        self.assertEqual(response.json()["error"]["code"], "document_ingestion_failed")


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class OneShotAnswerTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    @patch("qa.views.answer_questions")
    @patch("qa.views.ingest_document", return_value=8)
    def test_one_shot_pdf(self, mock_ingest, mock_answer):
        mock_answer.return_value = [
            {"question": "Q1?", "answer": "A1", "confidence": "high",
             "citations": [{"document_id": "x", "source": "doc.pdf", "excerpt": "..."}],
             "error": None},
        ]
        doc_file = SimpleUploadedFile("doc.pdf", b"%PDF-1.4 fake", content_type="application/pdf")
        questions_file = SimpleUploadedFile(
            "questions.json", json.dumps(["Q1?"]).encode(), content_type="application/json",
        )

        response = self.client.post(
            "/api/answer/",
            {"document_file": doc_file, "questions_file": questions_file},
            format="multipart",
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("document", data)
        self.assertIn("results", data)
        self.assertEqual(data["document"]["chunk_count"], 8)
        self.assertEqual(len(data["results"]), 1)
        mock_ingest.assert_called_once()
        mock_answer.assert_called_once()

    def test_one_shot_invalid_doc(self):
        doc_file = SimpleUploadedFile("doc.txt", b"plain", content_type="text/plain")
        questions_file = SimpleUploadedFile(
            "q.json", json.dumps(["Q?"]).encode(), content_type="application/json",
        )
        response = self.client.post(
            "/api/answer/",
            {"document_file": doc_file, "questions_file": questions_file},
            format="multipart",
        )
        self.assertEqual(response.status_code, 400)

    def test_one_shot_invalid_questions(self):
        doc_file = SimpleUploadedFile("doc.pdf", b"%PDF", content_type="application/pdf")
        questions_file = SimpleUploadedFile(
            "q.json", b"not valid json", content_type="application/json",
        )
        response = self.client.post(
            "/api/answer/",
            {"document_file": doc_file, "questions_file": questions_file},
            format="multipart",
        )
        self.assertEqual(response.status_code, 400)

    def test_one_shot_missing_files(self):
        response = self.client.post("/api/answer/", {}, format="multipart")
        self.assertEqual(response.status_code, 400)

    @patch("qa.views.ingest_document", side_effect=Exception("ingest boom"))
    def test_one_shot_ingest_failure(self, mock_ingest):
        doc_file = SimpleUploadedFile("doc.pdf", b"%PDF", content_type="application/pdf")
        questions_file = SimpleUploadedFile(
            "q.json", json.dumps(["Q?"]).encode(), content_type="application/json",
        )
        response = self.client.post(
            "/api/answer/",
            {"document_file": doc_file, "questions_file": questions_file},
            format="multipart",
        )
        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["error"]["code"], "document_ingestion_failed")


@override_settings(MEDIA_ROOT=TEST_MEDIA_ROOT)
class IndexViewTests(TestCase):
    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Zania Q&amp;A", html=True)
