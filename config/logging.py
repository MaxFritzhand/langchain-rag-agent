import json
import logging
from datetime import datetime, timezone

EXTRA_FIELDS = (
    "document_id", "duration_ms", "token_usage", "stage",
    "source_filename", "file_type", "size_kb", "chunks", "k", "model",
    "question", "answer_preview", "confidence", "sources", "scope",
    "request_id",
)


class JsonFormatter(logging.Formatter):
    """Structured JSON formatter — used in production / Docker."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in EXTRA_FIELDS:
            if hasattr(record, field):
                log_entry[field] = getattr(record, field)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


class PrettyFormatter(logging.Formatter):
    """Human-readable colored formatter — used in dev (DEBUG=True)."""

    LEVEL_COLOR = {
        "DEBUG": "\033[37m",     # gray
        "INFO": "\033[36m",      # cyan
        "WARNING": "\033[33m",   # yellow
        "ERROR": "\033[31m",     # red
        "CRITICAL": "\033[1;31m" # bold red
    }
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"

    def format(self, record):
        ts = datetime.now().strftime("%H:%M:%S")
        color = self.LEVEL_COLOR.get(record.levelname, "")
        level = f"{color}{record.levelname:5}{self.RESET}"

        stage = getattr(record, "stage", None)
        prefix = f"{self.BOLD}[{stage}]{self.RESET} " if stage else ""

        msg = record.getMessage()
        line = f"{self.DIM}{ts}{self.RESET} {level} {prefix}{msg}"

        kv_parts = []
        for field in EXTRA_FIELDS:
            if field == "stage":
                continue
            if hasattr(record, field):
                value = getattr(record, field)
                if isinstance(value, str) and len(value) > 80:
                    value = value[:77] + "..."
                kv_parts.append(f"{self.DIM}{field}={self.RESET}{value}")
        if kv_parts:
            line += "  " + " ".join(kv_parts)

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line
