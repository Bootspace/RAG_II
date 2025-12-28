from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

#Load embeddings and Vector store
persistent_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model= "text-embedding-3-small")
db= Chroma(
    persist_directory= persistent_directory,
    embedding_function= embedding_model,
    collection_metadata= { "hnsw:space":"cosine"}
)

model = ChatOpenAI(model="gpt-4o")

# Store our conversations as messages
chat_history = []

def ask_question(user_question):
    print(f"\n You asked: {user_question}")

    #Step 1- make the conversation clear using conversation history
    if chat_history:
        # Ask AI to make the question a standalone
        messages = [
            SystemMessage(content="Given the chathistory rewrite the question to be standalone")
        ] + chat_history  + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)
        search_question= result.content.strip()
        print(f" Searching for: {search_question}")

    else:
        search_question = user_question

    # Step 2 - Find relevant documents 
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents: ")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines= doc.page_content.split('\n')[:2]
        preview = "\n".join(lines)
        print(f"  Doc{i}: {preview}...")

    # step 3) Create final prompt
    combined_input = f"""Based on the following documents, please answer this question{ user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helper answer using only the information from these documents. If you can't find the answer in the document say "I don't have enough information to answer this question based on the provided documents"
    """

    # Step 4: get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history")
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

    result = model.invoke(messages)
    answer = result.content

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content= answer))

    print(f"Answer: {answer}")
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\n Your question: ")
        
        if question.lower() == 'quit':
            print("Goodbye")
            break

        ask_question(question)

if __name__== "__main__":
    start_chat()


