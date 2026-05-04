import json

from django.conf import settings
from rest_framework import serializers

from .models import Document

ALLOWED_FILE_TYPES = {"pdf", "json"}
MAX_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
MAX_QUESTIONS_FILE_SIZE_BYTES = settings.MAX_QUESTIONS_FILE_SIZE_MB * 1024 * 1024


def _file_extension(file_obj):
    return file_obj.name.rsplit(".", 1)[-1].lower() if "." in file_obj.name else ""


def _validate_document_file(value):
    ext = _file_extension(value)
    if ext not in ALLOWED_FILE_TYPES:
        raise serializers.ValidationError(
            f"Unsupported file type: .{ext}. Allowed: {', '.join(sorted(ALLOWED_FILE_TYPES))}"
        )
    if value.size > MAX_SIZE_BYTES:
        raise serializers.ValidationError(
            f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB"
        )
    return value


def _parse_questions_payload(parsed):
    if isinstance(parsed, dict):
        parsed = parsed.get("questions")

    if not isinstance(parsed, list) or not all(isinstance(q, str) for q in parsed):
        raise serializers.ValidationError("File must contain a JSON array of strings or an object with a questions array.")

    questions = [question.strip() for question in parsed if question.strip()]
    if not questions:
        raise serializers.ValidationError("At least one non-empty question is required.")
    if len(questions) > settings.MAX_QUESTIONS_PER_REQUEST:
        raise serializers.ValidationError(f"Maximum {settings.MAX_QUESTIONS_PER_REQUEST} questions allowed.")
    return questions


def _read_questions_file(file_obj):
    if _file_extension(file_obj) != "json":
        raise serializers.ValidationError("Questions file must be a .json file.")
    if file_obj.size > MAX_QUESTIONS_FILE_SIZE_BYTES:
        raise serializers.ValidationError(
            f"Questions file too large. Maximum size: {settings.MAX_QUESTIONS_FILE_SIZE_MB}MB"
        )

    try:
        content = file_obj.read().decode("utf-8")
        return _parse_questions_payload(json.loads(content))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise serializers.ValidationError("Invalid JSON file.")


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

    def validate_file(self, value):
        return _validate_document_file(value)


class DocumentResponseSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "filename", "file_type", "status", "chunk_count", "created_at"]


class QuestionsInputSerializer(serializers.Serializer):
    questions = serializers.ListField(
        child=serializers.CharField(max_length=1000),
        min_length=1,
        max_length=settings.MAX_QUESTIONS_PER_REQUEST,
        required=False,
    )
    questions_file = serializers.FileField(required=False)

    def validate_questions_file(self, value):
        return value

    def validate(self, data):
        has_questions = bool(data.get("questions"))
        has_file = bool(data.get("questions_file"))

        if not has_questions and not has_file:
            raise serializers.ValidationError(
                "Provide either 'questions' (JSON array) or 'questions_file' (JSON file)."
            )

        if has_file and not has_questions:
            try:
                data["questions"] = _read_questions_file(data["questions_file"])
            except serializers.ValidationError as exc:
                raise serializers.ValidationError({"questions_file": exc.detail})

        return data


class OneShotAnswerSerializer(serializers.Serializer):
    document_file = serializers.FileField()
    questions_file = serializers.FileField()

    def validate_document_file(self, value):
        return _validate_document_file(value)

    def validate_questions_file(self, value):
        try:
            self.context["questions"] = _read_questions_file(value)
        except serializers.ValidationError as exc:
            raise serializers.ValidationError(exc.detail)
        return value

    def validate(self, data):
        data["questions"] = self.context.get("questions", [])
        return data
