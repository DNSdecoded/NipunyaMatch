import fitz

MAX_PAGES = 20
MAX_BYTES = 10 * 1024 * 1024


class InvalidPDF(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def validate_pdf(data: bytes) -> fitz.Document:
    if len(data) >= MAX_BYTES:
        raise InvalidPDF("FILE_TOO_LARGE", "File is over 10 MB. Please upload a smaller PDF.")
    if not data.startswith(b"%PDF"):
        raise InvalidPDF("INVALID_PDF", "This file is not a PDF.")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:  # fitz raises generic errors on corrupt input
        raise InvalidPDF("INVALID_PDF", "This PDF could not be opened; it may be corrupt.") from e
    if doc.is_encrypted:
        raise InvalidPDF("INVALID_PDF", "This PDF is encrypted. Remove the password and retry.")
    if doc.page_count >= MAX_PAGES:
        raise InvalidPDF(
            "INVALID_PDF", f"This PDF has {doc.page_count} pages; the limit is {MAX_PAGES - 1}."
        )
    return doc
