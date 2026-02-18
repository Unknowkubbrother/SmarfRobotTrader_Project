from retrieval.chroma_client import ChromaDBClient

def main():
    print("🚀 Connecting to ChromaDB...")
    client = ChromaDBClient(persist_path="chroma_db")
    
    # 1. Get an existing ID (for example purposes)
    print("Fetching one ID to test...")
    existing_docs = client.collection.get(limit=4)
    
    if not existing_docs['ids']:
        print("❌ No documents found in DB. Run 'main.py' first!")
        return

    doc_id = existing_docs['ids'][1]
    print(f"🎯 Target ID: {doc_id}")

    # 2. Use get_document(id)
    print("\n--- Testing get_document(id) ---")
    doc_data = client.get_document(doc_id)
    
    if doc_data:
        print(f"✅ Document Found!")
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
        print("❌ Document not found via get_document()")

if __name__ == "__main__":
    main()
