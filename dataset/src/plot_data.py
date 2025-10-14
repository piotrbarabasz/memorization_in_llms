import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import Counter

DATASET_PATH = "C:/Users/user/Desktop/MSc/pb_msc/dataset/extracted_wikipedia_token_limit_records_5000.json"

def load_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_top_categories(data, top_n=10):
    output_file=f"C:/Users/user/Desktop/MSc/pb_msc/dataset/results/categories_distribiution_{top_n}_lowest.png"

    category_counts = Counter(item.get("category") for item in data if "category" in item)
    df_categories = pd.DataFrame(category_counts.items(), columns=["Category", "Count"])
    df_categories_sorted = df_categories.sort_values(by="Count", ascending=False).tail(top_n)

    plt.figure(figsize=(12, 8))

    plt.barh(df_categories_sorted["Category"], df_categories_sorted["Count"])
    plt.xlabel("Number of Articles")
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.title(f"Lowest {top_n} Categories in Dataset")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved to {output_file}")

def save_category_percentages(data, output_file="dataset/results/category_percentages.csv"):
    category_counts = Counter(item.get("category") for item in data if "category" in item)
    total = sum(category_counts.values())
    records = [
        {"Category": category, "Count": count, "Percentage": round((count / total) * 100, 2)}
        for category, count in category_counts.items()
    ]
    df = pd.DataFrame(records).sort_values(by="Percentage", ascending=False)
    df.to_csv(output_file, index=False)
    print(f"Category percentages saved to {output_file}")

def main():
    data = load_data(DATASET_PATH)
    
    # Plot top categories
    plot_top_categories(data, top_n=30)

    # Save category percentage CSV
    # save_category_percentages(data)

if __name__ == "__main__":
    main()
