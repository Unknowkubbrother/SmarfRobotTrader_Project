import chromadb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DB_PATH = "chroma_db"

def main():
    print(f"🚀 Connecting to ChromaDB at '{DB_PATH}'...")
    
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_or_create_collection(name="documents")
        
        count = collection.count()
        print(f"📊 Total Documents: {count}")
        
        if count == 0:
            print("❌ Database is empty.")
            return

        results = collection.get(
            limit=50,
            include=["metadatas", "documents"]
        )

        # 3. Format as Table
        data = []
        for i, doc_id in enumerate(results['ids']):
            item = {
                "id": doc_id,
                "document": results['documents'][i][:100] + "..." if len(results['documents'][i]) > 100 else results['documents'][i],
                **results['metadatas'][i]
            }
            data.append(item)
            
        df = pd.DataFrame(data)
        
        # Reorder columns for readability
        desired_order = ["symbol", "symbol_datetime", "timeframe", "document", "id"]
        cols = [c for c in desired_order if c in df.columns]
        remaining = [c for c in df.columns if c not in cols]
        df = df[cols + remaining]



        # 4. Display
        print("\n--- Recent 50 Documents ---")
        # Increase display width
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 50)
        print(df)
        
        print("\n💡 Tip: You can edit this script to export to CSV: df.to_csv('dump.csv')")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure run this script in the same folder as 'chroma_db'.")

if __name__ == "__main__":
    main()
