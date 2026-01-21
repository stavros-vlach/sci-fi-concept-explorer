import os
from dotenv import load_dotenv
import logging
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

logging.basicConfig(
    level=logging.INFO, filename="logfile", filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class SciFiQA:
    def __init__(self, model="gpt-4-turbo"):
        self.model = ChatOpenAI(
            model_name=model, 
            openai_api_base=os.getenv("OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.8)

    def set_data(self, data_path="Data Files"):
        loader = DirectoryLoader(data_path, show_progress=True)
        self.documents = loader.load()
    
    def split_data(self, documents, chunk_size=500, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                       chunk_overlap=chunk_overlap)
        self.split_documents = text_splitter.split_documents(self.documents)
        
    def create_embeddings(self, model="text-embedding-3-small"):
        embeddings_model = OpenAIEmbeddings(
            model=model,
            openai_api_base=os.getenv("OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.vector_store = Chroma.from_documents(
        documents=self.split_documents,
        embedding=embeddings_model,
        persist_directory="db/chroma_db"
        )   
        return self.vector_store
    
    def RAG_chain(self, question):
        system_message="You are a helpful creative assistant that gives ideas " \
                "to help writers create new Sci-Fi stories." \
                "You are given a context and a question"
        prompt = ChatPromptTemplate.from_template(
                """
                    Answer the question based only on the following context: {context}. Do not 
                    generate additional questions or answers.
                    Question: {question}
                """
            )
        consultant_messages = [SystemMessage(content=system_message),
                                    HumanMessage(content=question)]
        s_retriever = self.vector_store.as_retriever()
        retriever_1 = s_retriever

        chain_1 = (
                    {"context": retriever_1, "question": RunnablePassthrough()}
                    |prompt
                    |self.model
                    |StrOutputParser()
            )
        response_1 = chain_1.invoke(question)

        mmr_retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k":3})
        retriever_2 = mmr_retriever
            
        chain_2 = (
                    {"context": retriever_2, "question": RunnablePassthrough()}
                    |prompt
                    |self.model
                    |StrOutputParser()
            )
        response_2 = chain_2.invoke(question)

        return response_1, response_2


if __name__ == "__main__":
    controller = SciFiQA()
    logging.info(f"model: {controller.model.model_name} initialized successfully!")
    documents = controller.set_data()
    doc_chunks = controller.split_data(documents=documents)
    logging.info("Chunks from documents created successfully!")
    vector_store = controller.create_embeddings()
    logging.info("Vector store from documents created successfully!")
    query = "What are some common themes in Sci-Fi literature?"
    response_1, response_2 = controller.RAG_chain(question=query)
    logging.info(f"Response from model using standard retriever: {response_1}")
    logging.info(f"Response from model using MMR retriever: {response_2}")
