from transformers import BartTokenizer, BartForConditionalGeneration


model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

indian_legal_text = """
WHEREAS, in accordance with the laws of the Republic of India, this policy ("Policy") is established
to govern the conduct and responsibilities of individuals and entities engaging in commercial activities.

NOW, THEREFORE, be it resolved that:

1. COMPLIANCE WITH INDIAN LAW.

1.1 All parties subject to this Policy shall adhere to the laws and regulations of the Republic of India
    governing commercial activities, trade, and other relevant matters.

1.2 Non-compliance with Indian law may result in legal consequences, including fines, penalties,
    and legal proceedings.

2. DATA PROTECTION AND PRIVACY.

2.1 Parties shall handle personal data in accordance with the applicable data protection laws of India.

2.2 Consent of individuals shall be obtained before collecting, processing, or storing their personal information.

...

"""

inputs = tokenizer(indian_legal_text, return_tensors="pt", max_length=1024, truncation=True)

summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)


print("Original Indian Legal Policy Text:")
print(indian_legal_text)

print("\nGenerated Summary:")
print(summary)
