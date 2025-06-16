import os

def chunk_text(text, max_length=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        if end < len(text):
            last_period = text.rfind('.', start, end)
            if last_period != -1:
                end = last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks

def main():
    with open("website_data.txt", "r", encoding="utf-8") as f:
        full_text = f.read()
    chunks = chunk_text(full_text)
    os.makedirs("chunks", exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(f"chunks/chunk_{i}.txt", "w", encoding="utf-8") as f:
            f.write(chunk)
    print(f"Split text into {len(chunks)} chunks and saved in 'chunks/' folder")

if __name__ == "__main__":
    main()