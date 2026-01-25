import pdfplumber
import os

def load_pdf(file_path):
    """
    Extract text from a PDF file safely.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    return text


if __name__ == "__main__":
    pdf_path = os.path.join("data", "raw", "sample_document.pdf")

    try:
        extracted_text = load_pdf(pdf_path)

        with open(
            os.path.join("data", "processed", "cleaned_text.txt"),
            "w",
            encoding="utf-8"
        ) as file:
            file.write(extracted_text)

        print("✅ PDF text extraction completed successfully.")

    except Exception as e:
        print("❌ Error while processing PDF:")
        print(e)
