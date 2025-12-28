from langchain_text_splitters import RecursiveCharacterTextSplitter,CharacterTextSplitter

tesla_text = """Tesla's Q3 Results
This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

# spliter_1 = CharacterTextSplitter(
#     separator= "", # Default separator. Other options include ["\n\n", "\n", ". ", " ", ""]
#     chunk_size = 100,
#     chunk_overlap = 0
# )

# chunks_1 = spliter_1.split_text(tesla_text)

# for i, chunk in enumerate(chunks_1, 1):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(f'"{chunk}"')
#     print()

# Example 2: RecursiveCharacterTextSplitter fixes this
print("\n" + "=" * 60)
print("2. RECURSIVE CHARACTER TEXT SPLITTER SOLUTION")
print("=" * 60)

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " ", ""], # multiple separators in order of importance 
    chunk_size = 100,
    chunk_overlap = 0
)

chunks_2 = recursive_splitter.split_text(tesla_text)
print(f"Same problem text, but with RecursiveCharacterTextSplitter:")
for i, chunk in enumerate(chunks_2, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()