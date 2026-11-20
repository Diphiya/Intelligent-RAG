"""Typed exception hierarchy. Each error carries an HTTP status and a stable machine-readable code."""


class RAGError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str = ""):
        super().__init__(message or self.code)
        self.message = message or self.code


class UnsupportedFileType(RAGError):
    status_code, code = 415, "unsupported_file_type"


class FileTooLarge(RAGError):
    status_code, code = 413, "file_too_large"


class CorruptFile(RAGError):
    status_code, code = 422, "corrupt_file"


class EmptyDocument(RAGError):
    status_code, code = 422, "no_extractable_text"


class InvalidQuestion(RAGError):
    status_code, code = 400, "invalid_question"


class DocumentNotFound(RAGError):
    status_code, code = 404, "document_not_found"


class EmbeddingError(RAGError):
    status_code, code = 503, "embedding_unavailable"


class IndexMismatch(RAGError):
    status_code, code = 409, "index_embedder_mismatch"


class LLMUnavailable(RAGError):
    status_code, code = 503, "llm_unavailable"


class LLMBadResponse(RAGError):
    status_code, code = 502, "llm_bad_response"
