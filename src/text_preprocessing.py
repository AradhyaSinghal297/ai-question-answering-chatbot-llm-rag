def preprocess_text(text):
    """
    Clean and preprocess extracted text.
    """
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


if __name__ == "__main__":
    with open("data/processed/cleaned_text.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    cleaned_text = preprocess_text(raw_text)

    with open("data/processed/cleaned_text.txt", "w", encoding="utf-8") as file:
        file.write(cleaned_text)

    print("Text preprocessing completed successfully.")
