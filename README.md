#  Q&A Bot

A document question-answering service built with Django REST Framework, LangChain, Chroma, and OpenAI `gpt-4o-mini`.

The app accepts a PDF or JSON document plus a JSON file of questions, retrieves relevant document chunks, and returns structured question/answer pairs with citations.

## Architecture

```
Client / UI
   -> Django REST API
   -> LangChain RAG pipeline
      -> load PDF or JSON
      -> split into chunks
      -> embed with text-embedding-3-small
      -> store and query Chroma
      -> answer with gpt-4o-mini
```

Design choices aligned with the challenge rubric:

- Supports PDF and JSON document inputs.
- Supports multiple questions per request.
- Uses a two-step RAG chain for predictable latency and one LLM call per answered question.
- Applies relevance filtering before generation and returns a standard not-found answer when context is insufficient.
- Uses defensive prompting so retrieved document text is treated as untrusted data.
- Returns citations with source filename, chunk index, excerpt, and PDF page or JSON item index when available.
- Caps upload size, question-file size, questions per request, LLM timeout, and concurrent question answering.
- Includes Docker setup, structured JSON logs in production, and tests with mocked LLM/vector dependencies.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/answer/` | One-shot challenge endpoint: upload document and questions file, then receive answers |
| `POST` | `/api/documents/` | Upload and index a PDF or JSON document |
| `GET` | `/api/documents/{id}/` | Get document status and metadata |
| `POST` | `/api/documents/{id}/questions/` | Ask questions against one indexed document |
| `POST` | `/api/questions/` | Ask questions across all ready documents |
| `GET` | `/` | Minimal browser UI |

### One-Shot Challenge Endpoint

`POST /api/answer/` accepts multipart form data:

- `document_file`: required `.pdf` or `.json`
- `questions_file`: required `.json`

The questions file may be either:

```json
["Which cloud providers do you rely on?", "What are your SLAs for notification?"]
```

or:

```json
{"questions": ["Which cloud providers do you rely on?"]}
```

Example:

```bash
curl -X POST http://localhost:8000/api/answer/ \
  -F "document_file=@soc2-type2.pdf" \
  -F "questions_file=@questions.json"
```

Response shape:

```json
{
  "document": {
    "id": "7d62d2ea-5b4f-49ab-8570-708f3e0e82ff",
    "filename": "soc2-type2.pdf",
    "file_type": "pdf",
    "status": "ready",
    "chunk_count": 42,
    "created_at": "2026-05-04T12:00:00Z"
  },
  "results": [
    {
      "question": "Which cloud providers do you rely on?",
      "answer": "The document states that AWS is used as the cloud provider.",
      "confidence": "high",
      "citations": [
        {
          "document_id": "7d62d2ea-5b4f-49ab-8570-708f3e0e82ff",
          "source": "soc2-type2.pdf",
          "page": 6,
          "chunk_index": 2,
          "excerpt": "..."
        }
      ],
      "error": null
    }
  ]
}
```

## Local Setup

```bash
cd zania_qa
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=sk-your-key
DJANGO_SECRET_KEY=replace-with-a-long-random-secret
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1
```

Then run:

```bash
python manage.py migrate
python manage.py runserver
```

Open `http://localhost:8000`.

## Docker

```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY and DJANGO_SECRET_KEY
docker-compose up --build
```

The app runs at `http://localhost:8000`.

## Demo Script

Create a questions file:

```bash
cat > questions.json <<'JSON'
[
  "Do you have formally defined criteria for notifying a client during an incident?",
  "Which cloud providers do you rely on?",
  "Is personal information transmitted, processed, stored, or disclosed to third parties?"
]
JSON
```

Run the challenge flow:

```bash
curl -X POST http://localhost:8000/api/answer/ \
  -F "document_file=@soc2-type2.pdf" \
  -F "questions_file=@questions.json"
```

You can also use the browser UI’s "Challenge Flow" form to upload both files and display the answers.

## Tests

```bash
python manage.py test qa
python manage.py check
```

Tests mock LLM and vector-store calls, so they do not spend OpenAI credits.

`python manage.py check --deploy --fail-level WARNING` may report deployment warnings when running with local `DEBUG=True`, a development secret key, and no HTTPS proxy settings. For production/Docker deployments, set `DEBUG=False`, use a strong `DJANGO_SECRET_KEY`, restrict `ALLOWED_HOSTS`, and terminate HTTPS at the hosting layer.
If HTTPS is terminated directly by Django or a trusted proxy, also set `SECURE_HSTS_SECONDS`, `SECURE_HSTS_INCLUDE_SUBDOMAINS`, `SECURE_HSTS_PRELOAD`, `SECURE_SSL_REDIRECT`, `CSRF_COOKIE_SECURE`, and `SESSION_COOKIE_SECURE` appropriately for that environment.

## Configuration

| Variable or setting | Default | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | required | OpenAI API key loaded from `.env` or environment |
| `DJANGO_SECRET_KEY` | dev fallback | Set a long random value outside development |
| `DEBUG` | `True` | Use `False` for Docker/production |
| `ALLOWED_HOSTS` | `*` | Comma-separated Django hosts |
| `SECURE_HSTS_SECONDS` | `0` | HSTS duration for HTTPS deployments |
| `SECURE_HSTS_INCLUDE_SUBDOMAINS` | `False` | Apply HSTS to subdomains when safe |
| `SECURE_HSTS_PRELOAD` | `False` | Enable when the domain is ready for browser preload lists |
| `SECURE_SSL_REDIRECT` | `False` | Redirect HTTP to HTTPS when Django handles TLS policy |
| `CSRF_COOKIE_SECURE` | `False` | Mark CSRF cookies secure in HTTPS deployments |
| `SESSION_COOKIE_SECURE` | `False` | Mark session cookies secure in HTTPS deployments |
| `CHUNK_SIZE` | `1000` | Text splitter chunk size |
| `CHUNK_OVERLAP` | `200` | Text splitter overlap |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per question |
| `RETRIEVAL_FALLBACK_K` | `2` | Top chunks to keep when Chroma scores are low but candidates exist |
| `MIN_RELEVANCE_SCORE` | `0.2` | Minimum relevance score before calling the LLM |
| `MAX_QUESTIONS_PER_REQUEST` | `20` | Maximum questions per request |
| `MAX_CONCURRENT_QUESTIONS` | `3` | Worker cap for parallel question answering |
| `MAX_UPLOAD_SIZE_MB` | `50` | Maximum document upload size |
| `MAX_QUESTIONS_FILE_SIZE_MB` | `1` | Maximum questions file size |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-question LLM timeout |

## Repository Hygiene

Do not commit `.env`, `db.sqlite3`, `media/`, `chroma_db/`, or virtual environments. The included `.gitignore` and `.dockerignore` exclude local runtime artifacts and secrets.
