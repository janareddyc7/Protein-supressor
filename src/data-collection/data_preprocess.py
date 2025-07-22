import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Step 1: Load the dataset
print("Step 1: Loading the dataset...")
print("Current working directory:", os.getcwd())

# Adjust the path based on your directory structure
# From src/data-collection, go up one level, then to features
data_path = "../data/features/all_proteins_enhanced_mutations.csv"

try:
    df = pd.read_csv(data_path)
    print(f"✅ Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
except FileNotFoundError:
    print(f"❌ File not found at {data_path}")
    print("Please check the file path and try again.")
    exit()

print("\n" + "="*50)

# Step 2: Check for missing (NaN) values
print("Step 2: Checking for missing values...")
missing_values = df.isnull().sum()
total_missing = missing_values.sum()

print(f"Total missing values: {total_missing}")
if total_missing > 0:
    print("\nMissing values per column:")
    for col, missing in missing_values.items():
        if missing > 0:
            percentage = (missing / len(df)) * 100
            print(f"  {col}: {missing} ({percentage:.2f}%)")
else:
    print("✅ No missing values found!")

print("\n" + "="*50)

# Step 3: Drop non-useful columns
print("Step 3: Dropping non-useful columns...")
columns_to_drop = ["mutation_id", "wt_sequence", "mut_sequence"]

# Check which columns actually exist before dropping
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
missing_columns = [col for col in columns_to_drop if col not in df.columns]

if missing_columns:
    print(f"⚠️  Columns not found in dataset: {missing_columns}")

if existing_columns_to_drop:
    print(f"Dropping columns: {existing_columns_to_drop}")
    df_cleaned = df.drop(columns=existing_columns_to_drop)
    print(f"✅ Columns dropped successfully!")
    print(f"New dataset shape: {df_cleaned.shape}")
else:
    print("No columns to drop (they don't exist in the dataset)")
    df_cleaned = df.copy()

print("\n" + "="*50)

# Step 4: Verify ml_target column
print("Step 4: Verifying ml_target column...")

if 'ml_target' not in df_cleaned.columns:
    print("❌ ml_target column not found in dataset!")
    print(f"Available columns: {list(df_cleaned.columns)}")
    exit()

# Check if binary
unique_values = df_cleaned['ml_target'].unique()
print(f"Unique values in ml_target: {unique_values}")
print(f"Number of unique values: {len(unique_values)}")

# Check distribution
value_counts = df_cleaned['ml_target'].value_counts()
print(f"\nDistribution of ml_target:")
for value, count in value_counts.items():
    percentage = (count / len(df_cleaned)) * 100
    print(f"  {value}: {count} ({percentage:.2f}%)")

# Verify it's binary
if len(unique_values) == 2:
    print("✅ ml_target is binary!")
    
    # Check if reasonably balanced (neither class < 10%)
    min_percentage = min(value_counts) / len(df_cleaned) * 100
    if min_percentage >= 10:
        print(f"✅ Classes are reasonably balanced (smallest class: {min_percentage:.2f}%)")
    else:
        print(f"⚠️  Classes are imbalanced (smallest class: {min_percentage:.2f}%)")
else:
    print(f"❌ ml_target is not binary! Found {len(unique_values)} unique values.")

print("\n" + "="*50)

# Step 5: Split data into X and y
print("Step 5: Splitting data into features (X) and labels (y)...")

# Separate features and target
y = df_cleaned['ml_target']
X = df_cleaned.drop('ml_target', axis=1)

print(f"Features (X) shape: {X.shape}")
print(f"Labels (y) shape: {y.shape}")
print(f"Feature columns: {list(X.columns)}")

print("\n" + "="*50)

# Step 6: Stratified train-test split
print("Step 6: Performing stratified train-test split (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✅ Train-test split completed!")
print(f"Training set - X: {X_train.shape}, y: {y_train.shape}")
print(f"Test set - X: {X_test.shape}, y: {y_test.shape}")

# Verify stratification worked
print(f"\nClass distribution in training set:")
train_dist = y_train.value_counts(normalize=True)
for value, percentage in train_dist.items():
    print(f"  {value}: {percentage:.3f} ({percentage*100:.1f}%)")

print(f"\nClass distribution in test set:")
test_dist = y_test.value_counts(normalize=True)
for value, percentage in test_dist.items():
    print(f"  {value}: {percentage:.3f} ({percentage*100:.1f}%)")

print("\n" + "="*50)
print("🎉 Data preprocessing completed successfully!")
print("\nSummary:")
print(f"- Original dataset: {df.shape}")
print(f"- After cleaning: {df_cleaned.shape}")
print(f"- Features: {X.shape[1]}")
print(f"- Training samples: {X_train.shape[0]}")
print(f"- Test samples: {X_test.shape[0]}")

# Optional: Save the processed data
save_data = input("\nWould you like to save the processed data? (y/n): ").lower().strip()
if save_data == 'y':
    # Save to the same directory as the original file
    output_dir = "../data/features/"
    
    # Save training data
    X_train.to_csv(f"{output_dir}X_train.csv", index=False)
    y_train.to_csv(f"{output_dir}y_train.csv", index=False)
    
    # Save test data
    X_test.to_csv(f"{output_dir}X_test.csv", index=False)
    y_test.to_csv(f"{output_dir}y_test.csv", index=False)
    
    # Save cleaned full dataset
    df_cleaned.to_csv(f"{output_dir}cleaned_dataset.csv", index=False)
    
    print(f"✅ Processed data saved to {output_dir}")
    print("Files saved:")
    print("  - X_train.csv")
    print("  - y_train.csv") 
    print("  - X_test.csv")
    print("  - y_test.csv")
    print("  - cleaned_dataset.csv")
