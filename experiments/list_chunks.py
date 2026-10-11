from smartdoc.retrieval.vector import VectorStore


def main():
    chunks = VectorStore().get_all_documents()

    if not chunks:
        print("No chunks found. Index a document first.")
        return

    for chunk in chunks:
        print("=" * 60)
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Page: {chunk.metadata.get('page')}")
        print("-" * 60)
        print(chunk.page_content)
        print()


if __name__ == "__main__":
    main()