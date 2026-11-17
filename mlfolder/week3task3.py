import random
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# STEP 1: Generate 32 messy rows using a loop
# ---------------------------------------------------------
cities = [
    "Chennai",
    "chennai",
    " CHENNAI ",
    "Madurai",
    "madurai ",
    "Coimbatore",
    " SALEM",
]
rows = []

random.seed(42)  # Every time run pannumbodhum ore output vara

for i in range(1, 30):
    cust_id = 100 + i
    name = f"Customer_{i}"
    # Every 5th row + 29th row gets missing email (total 6 missing = 18.8%)
    email = None if (i % 5 == 0 or i == 29) else f"cust{i}@mail.com"
    city = cities[i % len(cities)]
    # 2 impossible ages (-5 and 250)
    age = -5 if i == 3 else (250 if i == 12 else random.randint(20, 55))
    # 2 wrong phone lengths
    phone = "98765" if i == 4 else ("123456789012" if i == 15 else f"98765432{i:02d}")

    rows.append([cust_id, name, email, city, age, phone])

# Add 3 duplicate rows using index, [1], [2] (Total = 32 rows)
rows.extend([rows[0], rows[1], rows[2]])

columns = ["customer_id", "name", "email", "city", "age", "phone"]
df_raw = pd.DataFrame(rows, columns=columns)
df_raw.to_csv("customers_raw.csv", index=False)

# ---------------------------------------------------------
# STEP 2: Data Audit (Before Cleaning)
# ---------------------------------------------------------
df = pd.read_csv("customers_raw.csv")
rows_before = len(df)

print("--- DATA AUDIT REPORT (BEFORE) ---")
missing_pct = (df.isna().mean() * 100).round(1)
print("\n% Missing per column:")
print(missing_pct)

dup_count = df.duplicated().sum()
invalid_age_count = ((df["age"] <= 0) | (df["age"] > 120)).sum()

print(f"\nNumber of duplicate rows: {dup_count}")
print(f"Count of invalid ages: {invalid_age_count}")

# ---------------------------------------------------------
# STEP 3: Data Cleaning
# ---------------------------------------------------------
df_clean = df.copy()

# 1. Trim spaces and standardise case
df_clean["city"] = df_clean["city"].str.strip().str.title()
df_clean["name"] = df_clean["name"].str.strip().str.title()

# 2. Remove duplicates
df_clean = df_clean.drop_duplicates()

# 3. Flag missing emails
df_clean["email_status"] = np.where(df_clean["email"].isna(), "Missing", "Valid")
df_clean["email"] = df_clean["email"].fillna("not_provided@flag.com")

# 4. Fix invalid ages using median
valid_median_age = df_clean[(df_clean["age"] > 0) & (df_clean["age"] <= 120)][
    "age"
].median()
invalid_age_mask = (df_clean["age"] <= 0) | (df_clean["age"] > 120)
ages_fixed_count = invalid_age_mask.sum()
df_clean.loc[invalid_age_mask, "age"] = int(valid_median_age)

# 5. Validate phone length (must be 10 digits)
df_clean["phone"] = df_clean["phone"].astype(str).str.strip()
df_clean["phone_valid"] = df_clean["phone"].str.len() == 10
df_clean = df_clean[df_clean["phone_valid"]].drop(columns=["phone_valid"])

# Save cleaned CSV
df_clean.to_csv("customers_clean.csv", index=False)

# ---------------------------------------------------------
# STEP 4: Before / After Summary
# ---------------------------------------------------------
rows_after = len(df_clean)
missing_email_before = missing_pct["email"]
missing_email_after = round(df_clean["email"].isna().mean() * 100, 1)

print("\n--- EXPECTED SUMMARY OUTPUT ---")
print(f"Rows before: {rows_before} | after: {rows_after}")
print(
    f"Missing email: {missing_email_before}% -> {int(missing_email_after)}% (flagged)"
)
print(f"Duplicates removed: {dup_count}")
print(f"Invalid ages fixed: {ages_fixed_count}")
