from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings.embeddings import Embeddings
import requests
import os
import httpx
import json
import time

class MistralRAGAgentRemote:
    def __init__(self, model="mistral-tiny"):
        self.API_KEY = os.environ['MISTRAL_API_KEY']
        self.pages = []
        self.vector_store = None
        self.embedder = CustomEmbedderMistral()
    def clear_vector_store(self):
        if self.vector_store:
            del self.vector_store
        self.pages = []
    def ingest_file(self, file_path, length=300, cross=50):
        loader = PyPDFLoader(file_path)
        for page in loader.lazy_load():
            self.pages.append(page)
    def prepare_all(self):
        self.vector_store = InMemoryVectorStore.from_documents(self.pages, self.embedder)
    def search(self, query):
        print("Searching in DB...")
        docs = self.vector_store.similarity_search_with_score(query, k=2)
        context = []
        for doc in docs:
            print(doc)
            if doc[1] >= 0.6:
                context.append(f'[[{doc[0].metadata["page"]}]]: {doc[0].page_content}\n')
        if len(context) == 0:
            return "No context specified!"
        print("Search done!")
        return "\n ------- \n".join(context)
    def respond(self, query):
        chat_template =f"You are a helpful AI assistant. Answer to any upcoming questions using only the given context, which you can ignore if the message is not a question. If possible also provide the page number on where the information is coming from, as specified in the special [[PAGE_NUMBER]] token. CONTEXT: {self.search(query)}"
        print(chat_template)
        headers = {
            "Authorization": f"Bearer {self.API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "mistral-medium",
            "messages": [
                {"role": "user", "content": chat_template},
                {"role": "user", "content": query}
            ],
            "stream": True
        }
        with httpx.stream("POST", "https://api.mistral.ai/v1/chat/completions", headers=headers, json=data) as response:
            for line in response.iter_lines():
                if line:
                    if line != "data: [DONE]":
                        try:
                            line_json = json.loads(line[5:])
                            yield line_json["choices"][0]["delta"]["content"] # Yield each token
                        except:
                            Exception("Something went wrong...")
        

class RAGAgent:
    def __init__(self, model="llama3.2"):
        self.pages = []
        self.embedder = OllamaEmbeddings(model=model)
        self.model = ChatOllama(model=model)
    def ingest_file(self, file_path):
        loader = PyPDFLoader(file_path)
        for page in loader.lazy_load():
            self.pages.append(page)
        
    def prepare_all(self):
        self.vector_store = InMemoryVectorStore.from_documents(self.pages, self.embedder)

    def search(self, query):
        print("Searching in DB...")
        docs = self.vector_store.max_marginal_relevance_search(query, k=2)
        context = []
        if len(docs) == 0:
            context.append("No context given.")
        else:
            for doc in docs:
                context.append(f'Page {doc.metadata["page"]}: {doc.page_content}\n')
        print("Search done!")
        return "\n ------- \n".join(context)
    
    def respond(self, query):
        chat_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful AI bot. You are to answer to questions (optimally with sources) from the user using only the following context: {context}."),
                ("human", "Hi, I have some questions."),
                ("ai", "Sure, let's get them answered!"),
                ("human", "{user_input}"),
            ]
        )
        message = chat_template.format(context=self.search(query), user_input=query)
        for chunk in self.model.stream(message):
            yield chunk.text()


class CustomEmbedderMistral(Embeddings):
    def __init__(self):
        self.documents = []
        self.API_KEY = os.environ["MISTRAL_API_KEY"]
    def embed_documents(self, texts):
        if len(texts) == 0:
            return []
        #Only do 10 pages at once
        chunk_size = 10
        embeddings = []
        for i in range(0, len(texts)//chunk_size+1):
            print(f"Currently at chunk {i}, {(i)*chunk_size}/{len(texts)} embedded.")
            texts_chunk = texts[i*chunk_size:(i+1)*chunk_size]
            if len(texts_chunk) == 0:
                break
            headers = {
                "Authorization": f"Bearer {self.API_KEY}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "mistral-embed",
                "input": texts_chunk
            }

            response = requests.post("https://api.mistral.ai/v1/embeddings", headers=headers, json=data).json()
            time.sleep(1.5)
            if "object" not in response:
                print(response)
            elif response["object"] == "error":
                print(response["message"])
                raise Exception("Something went wrong with embedding!")
            else:
                embeddings_chunk = [x["embedding"] for x in response["data"]]
                embeddings += embeddings_chunk
        return embeddings
        
    def embed_query(self, text):
        embedding = self.embed_documents([text])[0]
        return embedding
