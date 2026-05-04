import logging

from django.shortcuts import get_object_or_404, render
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Document
from .serializers import (
    DocumentResponseSerializer,
    DocumentUploadSerializer,
    OneShotAnswerSerializer,
    QuestionsInputSerializer,
)
from .services import answer_questions, ingest_document, reset_index

logger = logging.getLogger("qa")


class DocumentUploadView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data["file"]
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        size_kb = round(uploaded_file.size / 1024, 1)

        logger.info(
            f"POST /api/documents/  file={uploaded_file.name} ({size_kb} KB)",
            extra={"stage": "UPLOAD", "source_filename": uploaded_file.name, "file_type": ext, "size_kb": size_kb},
        )

        document = Document.objects.create(
            filename=uploaded_file.name,
            file_type=ext,
            file=uploaded_file,
        )

        try:
            chunk_count = ingest_document(document)
            document.status = "ready"
            document.chunk_count = chunk_count
            document.save()
        except Exception as e:
            logger.exception("Ingest failed", extra={"document_id": str(document.id)})
            document.status = "failed"
            document.save()
            return Response(
                {"error": f"Document ingestion failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        logger.info(
            f"Upload complete: {document.filename} -> {chunk_count} chunks",
            extra={"stage": "UPLOAD", "document_id": str(document.id), "chunks": chunk_count},
        )
        return Response(
            DocumentResponseSerializer(document).data,
            status=status.HTTP_201_CREATED,
        )


class DocumentDetailView(APIView):
    def get(self, request, document_id):
        document = get_object_or_404(Document, id=document_id)
        return Response(DocumentResponseSerializer(document).data)


class OneShotAnswerView(APIView):
    """Upload a document and questions file in one request, then return answers."""

    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        serializer = OneShotAnswerSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        uploaded_file = serializer.validated_data["document_file"]
        questions = serializer.validated_data["questions"]
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        size_kb = round(uploaded_file.size / 1024, 1)

        logger.info(
            f"POST /api/answer/  doc={uploaded_file.name} ({size_kb} KB), questions={len(questions)}",
            extra={
                "stage": "ONE-SHOT",
                "source_filename": uploaded_file.name,
                "file_type": ext,
                "size_kb": size_kb,
                "chunks": len(questions),
            },
        )

        document = Document.objects.create(
            filename=uploaded_file.name,
            file_type=ext,
            file=uploaded_file,
        )

        try:
            chunk_count = ingest_document(document)
            document.status = "ready"
            document.chunk_count = chunk_count
            document.save()
        except Exception as e:
            logger.exception("One-shot ingest failed", extra={"document_id": str(document.id)})
            document.status = "failed"
            document.save()
            return Response(
                {
                    "error": {
                        "code": "document_ingestion_failed",
                        "message": f"Document ingestion failed: {str(e)}",
                    },
                    "document": DocumentResponseSerializer(document).data,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        results = answer_questions(questions, document_id=str(document.id))
        return Response(
            {
                "document": DocumentResponseSerializer(document).data,
                "results": results,
            },
            status=status.HTTP_200_OK,
        )


class QuestionsView(APIView):
    """Ask questions scoped to a single document."""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, document_id):
        document = get_object_or_404(Document, id=document_id)

        if document.status != "ready":
            return Response(
                {"error": f"Document is not ready. Current status: {document.status}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = QuestionsInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        questions = serializer.validated_data["questions"]
        logger.info(
            f"POST /api/documents/{str(document.id)[:8]}.../questions/  count={len(questions)}",
            extra={"stage": "REQUEST", "document_id": str(document.id), "chunks": len(questions)},
        )
        results = answer_questions(questions, document_id=str(document.id))

        return Response(results)


class AllDocumentsQuestionsView(APIView):
    """Ask questions across all uploaded documents."""

    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        if not Document.objects.filter(status="ready").exists():
            return Response(
                {"error": "No documents have been indexed yet."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = QuestionsInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        questions = serializer.validated_data["questions"]
        logger.info(
            f"POST /api/questions/  count={len(questions)} scope=all-docs",
            extra={"stage": "REQUEST", "scope": "all-docs", "chunks": len(questions)},
        )
        results = answer_questions(questions)

        return Response(results)


class ResetIndexView(APIView):
    """Clear local uploaded documents and vector index for development/demo resets."""

    def post(self, request):
        result = reset_index()
        return Response(result)


def index_view(request):
    documents = Document.objects.all().order_by("-created_at")[:10]
    ready_document_count = Document.objects.filter(status="ready").count()
    return render(
        request,
        "qa/index.html",
        {
            "documents": documents,
            "ready_document_count": ready_document_count,
        },
    )
