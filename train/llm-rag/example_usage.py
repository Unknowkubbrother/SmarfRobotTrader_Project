from retrieval.chroma_client import ChromaDBClient

def main():
    print("🚀 Connecting to ChromaDB...")
    client = ChromaDBClient(persist_path="chroma_db")
    
    # 1. Get ALL documents (remove limit=4)
    print("Fetching ALL documents...")
    existing_docs = client.collection.get() # No limit means get ALL
    
    count = len(existing_docs['ids'])
    print(f"✅ Found {count} documents in Total.")
    
    if count == 0:
        print("❌ No documents found in DB. Run 'main.py' first!")
        return

    # Iterate over ALL documents found
    print(f"\n--- Detailed View of All {count} Documents ---")
    for i, doc_id in enumerate(existing_docs['ids']):
        print(f"\n--- Document {i+1} ---")
        doc_data = client.get_document(doc_id)
        
        if doc_data:
            print(f"ID: {doc_data['id']}")
            print(f"Symbol: {doc_data['metadata']['symbol']}")
            print(f"Symbol Datetime: {doc_data['metadata']['symbol_datetime']}")
            print(f"Timeframe: {doc_data['metadata']['timeframe']}")
            print(f"Content Preview: {doc_data['content'][:50]}...")
            
            # Access the Embedding Vector
            vector = doc_data['embedding']
            print(f"Vector Length: {len(vector)}")
            print(f"Vector Preview: {vector[:5]}...")  # Show first 5 numbers
        else:
            print(f"❌ Document {doc_id} not found via get_document()")

if __name__ == "__main__":
    main()
