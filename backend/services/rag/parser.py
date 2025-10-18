# File: backend/services/rag/parser.py
import fitz  # PyMuPDF
import io

def parse_pdf(file_contents: bytes) -> str:
    """
    Membaca konten byte dari file PDF dan mengekstrak semua teks.

    Args:
        file_contents: Konten file PDF dalam bentuk bytes.

    Returns:
        String berisi seluruh teks yang diekstrak dari PDF.
    """
    text_content = ""
    try:
        # Membuka PDF dari memory (bytes)
        pdf_document = fitz.open(stream=file_contents, filetype="pdf")

        # Iterasi melalui setiap halaman
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text_content += page.get_text() # Ekstrak teks dari halaman

        pdf_document.close()
        return text_content

    except Exception as e:
        print(f"Error saat parsing PDF: {e}")
        return f"Error: Gagal memproses file PDF. Detail: {str(e)}"