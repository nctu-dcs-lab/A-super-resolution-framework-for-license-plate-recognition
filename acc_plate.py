import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv("PKU_VGG_CRNN.csv")

# Exclude the last row (Average row)
df = df[:-1]

# Calculate the total number of plates
total_plates = df.shape[0]

# Calculate the number of plates with accuracy >= 5
plates_accuracy_5_or_higher = df[df['Accuracy'] >= 5]
num_plates_accuracy_5_or_higher = plates_accuracy_5_or_higher.shape[0]

# Calculate the number of plates with accuracy >= 6
plates_accuracy_6_or_higher = df[df['Accuracy'] >= 6]
num_plates_accuracy_6_or_higher = plates_accuracy_6_or_higher.shape[0]

# Calculate the number of plates with accuracy == 7
plates_accuracy_7 = df[df['Accuracy'] == 7]
num_plates_accuracy_7 = plates_accuracy_7.shape[0]

# Calculate the percentage of plates with accuracy >= 5
percentage_accuracy_5_or_higher = (num_plates_accuracy_5_or_higher / total_plates) * 100

# Calculate the percentage of plates with accuracy >= 6
percentage_accuracy_6_or_higher = (num_plates_accuracy_6_or_higher / total_plates) * 100

# Calculate the percentage of plates with accuracy == 7
percentage_accuracy_7 = (num_plates_accuracy_7 / total_plates) * 100

print(f"Total number of plates: {total_plates}")
print(f"Number of plates with accuracy >= 5: {num_plates_accuracy_5_or_higher}")
print(f"Number of plates with accuracy >= 6: {num_plates_accuracy_6_or_higher}")
print(f"Number of plates with accuracy == 7: {num_plates_accuracy_7}")
print(f"Percentage of plates with accuracy >= 5: {percentage_accuracy_5_or_higher:.2f}%")
print(f"Percentage of plates with accuracy >= 6: {percentage_accuracy_6_or_higher:.2f}%")
print(f"Percentage of plates with accuracy == 7: {percentage_accuracy_7:.2f}%")
