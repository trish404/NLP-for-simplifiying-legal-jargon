from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Sample legal text
legal_text = """
THIS AGREEMENT (the "Agreement") is entered into on this 1st day of January 2024,
by and between Party A, a corporation organized and existing under the laws of the State of XYZ,
and Party B, a limited liability company organized and existing under the laws of the State of ABC.

RECITALS:

WHEREAS, Party A desires to engage Party B to provide consulting services related to financial analysis;
WHEREAS, Party B agrees to provide such services on the terms and conditions set forth in this Agreement.

NOW, THEREFORE, in consideration of the mutual covenants contained herein, the parties agree as follows:

1. SERVICES.

1.1 Engagement of Services. Party B agrees to provide financial analysis consulting services
   to Party A in accordance with the terms of this Agreement.

1.2 Scope of Services. The scope of services to be provided by Party B shall include but not be limited to
   financial modeling, risk assessment, and market trend analysis.

2. COMPENSATION.

2.1 Payment. In consideration for the services provided by Party B, Party A agrees to pay
   Party B a fee of $10,000 per month.

2.2 Invoicing. Party B shall submit monthly invoices for services rendered, and payment shall be due
   within 30 days of receipt of the invoice.

...

"""

# Tokenize and convert the text to tensor
inputs = tokenizer(legal_text, return_tensors="pt", max_length=1024, truncation=True)

# Generate summary using BART model
summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Display the original text and the generated summary
print("Original Text:")
print(legal_text)

print("\nGenerated Summary:")
print(summary)
