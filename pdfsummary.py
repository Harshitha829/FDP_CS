from PyPDF2 import PdfReader
from transformers import pipeline

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to summarize text
def summarize_text(text, max_length=150, min_length=40):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Truncate input to fit model's max token limit
    text = text[:1024]  # For larger input, you'd split into chunks
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example usage
pdf_path = "C:/Users/Lenovo21/Desktop/genaai/Programs/Python/artificial_intelligence_tutorial.pdf"  # Update path
pdf_text = extract_text_from_pdf(pdf_path)
summary = summarize_text(pdf_text)

print("\n📄 Summary:\n", summary)
