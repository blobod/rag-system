from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

print("Loading Mistral-7B")

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

print("Model loaded now Testing generation")

prompt = "<s>[INST] What is machine learning? Answer in one sentence. [/INST]"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"\nResponse: {response}")
print("\nTest complete")