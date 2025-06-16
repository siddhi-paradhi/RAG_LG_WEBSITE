import fitz  
import os

def extract_all_text(pdf_folder="data"):
    full_text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(pdf_folder, filename))
            for page in doc:
                full_text += page.get_text()
            doc.close()
    with open("website_data.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print("Extracted all PDF text to 'website_data.txt'")

if __name__ == "__main__":
    extract_all_text()