import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from pathlib import Path

class VectorDbManager:
    __client: QdrantClient
    __dense_embeddings: HuggingFaceEmbeddings
    __sparse_embeddings: FastEmbedSparse
    def __init__(self):
        try:
            self.__client = QdrantClient(path=config.QDRANT_DB_PATH)
        except RuntimeError as e:
            msg = str(e)
            if "already accessed by another instance" in msg:
                lock_path = Path(config.QDRANT_DB_PATH) / ".lock"
                raise RuntimeError(
                    "Local Qdrant storage is locked.\n\n"
                    f"Path: {config.QDRANT_DB_PATH}\n"
                    f"Lock file: {lock_path}\n\n"
                    "Fix options:\n"
                    "1) Stop the other running app/notebook using this same Qdrant folder.\n"
                    "2) If nothing is running, delete the lock file.\n"
                    "3) Or run this app with a different storage path, e.g.\n"
                    "   `QDRANT_DB_PATH=qdrant_db_2 python project/app.py`\n\n"
                    f"Original error: {msg}"
                ) from e
            raise
        self.__dense_embeddings = HuggingFaceEmbeddings(model_name=config.DENSE_MODEL)
        self.__sparse_embeddings = FastEmbedSparse(model_name=config.SPARSE_MODEL)

    def create_collection(self, collection_name):
        if not self.__client.collection_exists(collection_name):
            print(f"Creating collection: {collection_name}...")
            self.__client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=len(self.__dense_embeddings.embed_query("test")), distance=qmodels.Distance.COSINE),
                sparse_vectors_config={config.SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
            )
            print(f"✓ Collection created: {collection_name}")
        else:
            print(f"✓ Collection already exists: {collection_name}")

    def delete_collection(self, collection_name):
        try:
            if self.__client.collection_exists(collection_name):
                print(f"Removing existing Qdrant collection: {collection_name}")
                self.__client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: could not delete collection {collection_name}: {e}")

    def get_collection(self, collection_name) -> QdrantVectorStore:
        try:
            return QdrantVectorStore(
                    client=self.__client,
                    collection_name=collection_name,
                    embedding=self.__dense_embeddings,
                    sparse_embedding=self.__sparse_embeddings,
                    retrieval_mode=RetrievalMode.HYBRID,
                    sparse_vector_name=config.SPARSE_VECTOR_NAME
                )
        except Exception as e:
            print(f"Unable to get collection {collection_name}: {e}")
