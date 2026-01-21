import pytest
from first_assigment.application import SciFiQA
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

def test_set_data():
    app = SciFiQA()
    
    documents = [
                   Document(page_content="This is a long document about sci-fi."*50,
                         metadata={"source": "test_source"})
    ]

    app.documents = documents
    app.split_data(documents, chunk_size=50, chunk_overlap=10)
    assert len(app.split_documents) > 1
    for doc in app.split_documents:
        assert len(doc.page_content) <= 50

def test_create_embeddings(tmp_path):
    app = SciFiQA()

    documents = [
        Document(page_content="This is a test.",
                 metadata={"source": "test"})
    ]
    app.documents = documents
    app.split_data(documents, chunk_size=50, chunk_overlap=10)

    vector_store = app.create_embeddings(model="text-embedding-3-small")

    assert len(vector_store) > 0

def test_RAG_chain_responses():
    app = SciFiQA()

    documents = [
        Document(page_content="This is a test.",
                 metadata={"source": "test"})
    ]
    app.documents = documents
    app.split_data(documents, chunk_size=50, chunk_overlap=10)
    app.create_embeddings(model="text-embedding-3-small")

    question = "What is a sci-fi story?"
    response_1, _ = app.RAG_chain(question)

    assert len(response_1) > 0
    