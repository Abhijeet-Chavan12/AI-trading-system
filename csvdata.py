import pandas as pd

# Load all tables from the Wikipedia page
url = "https://en.wikipedia.org/wiki/NIFTY_50"
tables = pd.read_html(url)
print(f"Found {len(tables)} tables")

# The table at index 1 contains NIFTY 50 companies
nifty_table = tables[1]
print(nifty_table.head())  # Just to confirm the format

# Use the correct column name casing
nifty_table = nifty_table[["Company name", "Symbol"]]
nifty_table["Symbol"] = nifty_table["Symbol"].apply(lambda x: x + ".NS")

# Save to CSV
nifty_table.to_csv("nse_symbols.csv", index=False)
print("✅ Saved as 'nse_symbols.csv'")
