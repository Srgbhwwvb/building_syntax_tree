from datasets import load_dataset

dataset = load_dataset("Tonic/MiniF2F")


print(dataset)
print(dataset.shape)
print(dataset.keys())
print(dataset['train'][0].keys())

for i in range(3):  # Первые 3 примера
    example = dataset['train'][i]
    print(f"\nПример {i+1}:")
    print(f"Название: {example['name']}")
    print(f"Неформальное условие:\n{example['informal_prefix']}")
    print(f"Формальное представление:\n{example['formal_statement']}\n{'-'*50}")