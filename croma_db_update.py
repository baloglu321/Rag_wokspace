import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.errors import NotFoundError
import hashlib
import os

# --- Ayarlar ---
CHROMA_HOST = "localhost" # Sadece ana bilgisayar adı
CHROMA_PORT = 8000        # Sadece port numarası
COLLECTION_NAME = "gora_arog_rag_koleksiyonu"
FILE_PATH = "/home/mbaloglu/Rag/database/" # Kendi dosyanızın adı


def db_update():
   
    # 4. ChromaDB İstemcisi Oluşturma ve Bağlanma
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    print(f"Eski koleksiyon ({COLLECTION_NAME}) siliniyor...")
    try:
            # Silme işlemini dene
            client.delete_collection(COLLECTION_NAME) 
            print("Silme başarılı.")
    except NotFoundError:
            # Eğer koleksiyon yoksa (ilk çalıştırma), hatayı yoksay ve devam et
            print("Koleksiyon zaten mevcut değil. Yeni koleksiyon oluşturulacak.")
    # 5. Chroma Veri Deposu Oluşturma (Bağlantılı)
    # Vektörler bu koleksiyona yüklenir.
    vectorstore=update_db_with_feedback(FILE_PATH,client=client,collection_name=COLLECTION_NAME)
        
    print(f"Vektörler başarıyla Docker sunucusundaki '{COLLECTION_NAME}' koleksiyonuna yüklendi.")
    # 5. Sorgu Hazırlama (Retrieval)
    return vectorstore

def create_id(chunk_content):
    """Metin içeriğinden kalıcı bir SHA256 hash ID'si oluşturur."""
    return hashlib.sha256(chunk_content.encode('utf-8')).hexdigest()


def get_chunks_with_ids(file_dir_path: str, chunk_size: int = 2000, chunk_overlap: int = 200):
    """Veri dizinindeki tüm dosyaları yükler, parçalar ve her parçaya içerik tabanlı ID atar."""
    
    files = os.listdir(file_dir_path)
    all_documents = []
    
    for f in files:
        file = os.path.join(file_dir_path, f)
        try:
            loader = TextLoader(file, encoding="utf8")
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"UYARI: {f} yüklenirken hata oluştu: {e}")
            continue

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", " ", ""]
    )
    all_chunks = text_splitter.split_documents(all_documents)
    
    ids = []
    for chunk in all_chunks:
        # Metin içeriğine bağlı ID oluşturma
        chunk_id = hashlib.sha256(chunk.page_content.encode('utf-8')).hexdigest()
        ids.append(chunk_id)

    return all_chunks, ids

def update_db_with_feedback(file_dir_path: str, client: chromadb.HttpClient, collection_name: str):
    """
    Veritabanını günceller, mevcut parçaları kontrol eder ve geri bildirim sağlar.
    (Mevcut parçalar overwrite edilir, silinmez).
    """
    
    # Veriyi hazırlama
    all_chunks, ids = get_chunks_with_ids(file_dir_path)
    print(f"Hazırlanan toplam parça sayısı: {len(all_chunks)}")

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vektör deposunu al (Eğer koleksiyon yoksa oluşturur)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    
    # ------------------ DURUM KONTROLÜ VE YÜKLEME ------------------
    
    # ChromaDB'ye parça parça ekleme/güncelleme (Daha fazla kontrol için)
    # LangChain'in add_documents'u arka planda ID'ye göre güncelleme (overwrite) yapar.
    
    # Mevcut tüm ID'leri çekelim (Sadece durum kontrolü için)
    try:
        existing_ids = client.get_collection(collection_name).get(include=[])['ids']
        existing_ids_set = set(existing_ids)
    except Exception:
        # Koleksiyon ilk kez oluşturuluyorsa
        existing_ids_set = set()
    
    chunks_to_add = []
    ids_to_add = []
    
    # Hangi parçaların yeni, hangilerinin güncelleneceğini belirleme (loglama için)
    new_count = 0
    updated_count = 0
    
    for i, chunk_id in enumerate(ids):
        if chunk_id in existing_ids_set:
            # ID zaten var, bu OVERWRITE (güncelleme) olacak
            updated_count += 1
            # Güncelleme de ekleme işlemiyle aynıdır
            chunks_to_add.append(all_chunks[i])
            ids_to_add.append(chunk_id)
        else:
            # ID yok, bu yeni bir ekleme olacak
            new_count += 1
            chunks_to_add.append(all_chunks[i])
            ids_to_add.append(chunk_id)

    # Yükleme (LangChain'in add_documents metodu hem yeni ekler hem de var olan ID'leri günceller)
    if chunks_to_add:
        # Bu işlem arka planda hem yeni ekler hem de eski ID'leri günceller (overwrite)
        vectorstore.add_documents(documents=chunks_to_add, ids=ids_to_add)

    print(f"\n--- Yükleme Özeti ---")
    print(f"Toplam Parça İşlendi: {len(all_chunks)}")
    print(f"✅ Yeni Eklendi: {new_count} adet parça.")
    print(f"🔄 Güncellendi (Overwrite Edildi): {updated_count} adet parça.")
    print("Veritabanı başarıyla güncellendi.")
    
    return vectorstore



def test(vectorstore):
    query = "Cem Yılmaz'ın oynadığı karakterin logar ile ilgili ilginç bir sözü neydi?"

    # En alakalı 3 adet parçayı (chunk) getir.
    retrieved_docs = vectorstore.similarity_search(query, k=3)

    print(f"\nSorgu: '{query}'")
    print("-" * 40)
    print(f"ChromaDB'den Gelen {len(retrieved_docs)} En Alakalı Parça:")

    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Parça {i+1} ---")
        print(f"Kaynak: {doc.metadata.get('source', 'Bilinmiyor')}")
        print(doc.page_content[:250] + "...")

if __name__ == "__main__":
    vectorstore=db_update()
    
