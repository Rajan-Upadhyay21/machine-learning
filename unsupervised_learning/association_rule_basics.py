transactions = [
    ["Milk", "Bread", "Butter"],
    ["Bread", "Butter"],
    ["Milk", "Bread"],
    ["Milk", "Butter"],
    ["Bread", "Eggs"],
    ["Milk", "Bread", "Eggs"],
    ["Butter", "Eggs"]
]

item_counts = {}

for transaction in transactions:
    for item in transaction:
        item_counts[item] = item_counts.get(item, 0) + 1

print("Item Frequency Counts:")
for item, count in item_counts.items():
    print(f"{item}: {count}")
