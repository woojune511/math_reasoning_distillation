from datasets import load_dataset

def inspect():
    print("Loading dataset...")
    ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train")
    print("Columns:", ds.column_names)
    print("First example:", ds[0])

if __name__ == "__main__":
    inspect()
