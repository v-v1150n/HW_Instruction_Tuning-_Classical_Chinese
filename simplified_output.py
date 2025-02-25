import json

with open('prediction.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    translated_text = item['output'].split("翻譯: ")[-1]
    item['output'] = translated_text.strip()

with open('simplified_prediction.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=2)

print("The simplified output has been saved to simplified_prediction.json")
