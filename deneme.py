from croma_db_update import update_db_with_feedback
import chromadb
CHROMA_HOST = "localhost" # Sadece ana bilgisayar adı
CHROMA_PORT = 8000        # Sadece port numarası
COLLECTION_NAME = "gora_arog_rag_koleksiyonu"
client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
update_db_with_feedback("/home/mbaloglu/Rag/database-2",client=client,collection_name="gora_arog_rag_koleksiyonu")
update_db_with_feedback("/home/mbaloglu/Rag/database",client=client,collection_name="gora_arog_rag_koleksiyonu")