from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to read a text file and return content
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

# Function to summarize the content
def summarize_text(text, max_length=130, min_length=30):
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

# Example usage
file_path = "C:/Users/Lenovo21/Desktop/genaai/Programs/Python/ADA.txt"
# Replace with your text file path
text = read_text_file(file_path)

# BART has a token limit (~1024), truncate if needed
text = text[:1000]  

summary = summarize_text(text)
print("\n📄 Summary:\n", summary)
