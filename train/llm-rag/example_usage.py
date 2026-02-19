from retrieval.chroma_client import ChromaDBClient

def main():
    print("🚀 Connecting to ChromaDB...")
    client = ChromaDBClient(persist_path="chroma_db")
    
    print("Fetching ALL documents...")
    existing_docs = client.collection.get()
    
    count = len(existing_docs['ids'])
    print(f"✅ Found {count} documents in Total.")
    
    if count == 0:
        print("❌ No documents found in DB. Run 'main.py' first!")
        return

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
            
            vector = doc_data['embedding']
            print(f"Vector Length: {len(vector)}")
            print(f"Vector Preview: {vector[:5]}...")
        else:
            print(f"❌ Document {doc_id} not found via get_document()")

if __name__ == "__main__":
    main()
