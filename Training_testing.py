import pandas as pd

# -------------------------------
# Create Table Data
# -------------------------------
data = {
    "Class label": ["Metal", "Metal", "Glass", "Plastic", "Trash"],
    "Training": [329, 329, 401, 386, 110],
    "Testing": [83, 83, 111, 98, 29],
    "Total": [412, 412, 512, 484, 139]
}

df = pd.DataFrame(data)

# Add Total Row
total_row = pd.DataFrame({
    "Class label": ["Total"],
    "Training": [df["Training"].sum()],
    "Testing": [df["Testing"].sum()],
    "Total": [df["Total"].sum()]
})

df = pd.concat([df, total_row], ignore_index=True)

# Display Table
print("\nTable 4: Sample Class Distribution on East Waste\n")
print(df)