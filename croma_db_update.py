import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.errors import NotFoundError
import hashlib
import os
import json

# --- Ayarlar ---
CHROMA_HOST = "localhost"  # Sadece ana bilgisayar adı
CHROMA_PORT = 8000  # Sadece port numarası
COLLECTION_NAME = "rag_test_data"
FILE_PATH = "/home/mbaloglu/Rag/database/"  # Kendi dosyanızın adı
MAX_BATCH_SIZE = 5000


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
    vectorstore = update_db_with_feedback(
        FILE_PATH, client=client, collection_name=COLLECTION_NAME
    )

    print(
        f"Vektörler başarıyla Docker sunucusundaki '{COLLECTION_NAME}' koleksiyonuna yüklendi."
    )
    # 5. Sorgu Hazırlama (Retrieval)
    return vectorstore


def create_id(chunk):
    """Metin içeriği, kaynak dosyası ve metin sırasını kullanarak benzersiz ID oluşturur."""
    # Kaynak dosyası (SQuAD verisinde: dosya adı + başlık)
    source = chunk.metadata.get("source", "") + chunk.metadata.get("title", "")

    # Metin içeriği
    content = chunk.page_content

    # Metin + Kaynak bilgisini birleştirip hash alıyoruz
    unique_string = f"{source}_{content}"

    return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()


def get_chunks_with_ids(
    file_dir_path: str, chunk_size: int = 500, chunk_overlap: int = 50
):
    """Veri dizinindeki tüm JSON dosyalarını okur, SQuAD formatını çözer ve parçalara ayırır."""

    all_documents = []

    for filename in os.listdir(file_dir_path):
        if filename.endswith(".json"):
            file_path = os.path.join(file_dir_path, filename)
            print(f"-> {filename} dosyası okunuyor...")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # SQuAD JSON yapısını gezme
            # Yapı: data -> paragraphs -> context
            for item in data.get("data", []):
                for paragraph_data in item.get("paragraphs", []):
                    # Her bir paragrafı (context), tüm SQuAD veri setinin ana bağlamını temsil eden
                    # tek bir belge olarak alıyoruz.
                    context_text = paragraph_data.get("context")
                    title = item.get("title", "Bilinmeyen Başlık")

                    if context_text:
                        # LangChain Document nesnesi oluşturma
                        all_documents.append(
                            Document(
                                page_content=context_text,
                                metadata={"source": file_path, "title": title},
                            )
                        )

    # 2. Bölme (Chunking) - Artık parçalama işlemini büyük paragraflar üzerinde yapıyoruz
    print(
        f"Toplam {len(all_documents)} paragraf/belge hazırlandı. Parçalara ayrılıyor..."
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    all_chunks = text_splitter.split_documents(all_documents)

    # 3. ID Oluşturma ve Ayırma
    ids = []
    for i, chunk in enumerate(all_chunks):  # enumerate ile sırasını (i) alıyoruz
        # Metin içeriğine sadece i (index) ekleyerek benzersizliği garanti ediyoruz
        unique_string = f"{chunk.page_content}_{i}"
        chunk_id = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()

        ids.append(chunk_id)
        chunk.metadata["ids"] = chunk_id

    return all_chunks, ids


def update_db_with_feedback(
    file_dir_path: str, client: chromadb.HttpClient, collection_name: str
):
    """
    Veritabanını günceller, mevcut parçaları kontrol eder ve geri bildirim sağlar.
    (Mevcut parçalar overwrite edilir, silinmez).
    """

    # Veriyi hazırlama
    all_chunks, ids = get_chunks_with_ids(file_dir_path)
    print(f"Hazırlanan toplam parça sayısı: {len(all_chunks)}")

    embedding_function = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )

    # Vektör deposunu al (Eğer koleksiyon yoksa oluşturur)
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function,
    )

    # ------------------ DURUM KONTROLÜ VE YÜKLEME ------------------

    # ChromaDB'ye parça parça ekleme/güncelleme (Daha fazla kontrol için)
    # LangChain'in add_documents'u arka planda ID'ye göre güncelleme (overwrite) yapar.

    # Mevcut tüm ID'leri çekelim (Sadece durum kontrolü için)
    try:
        existing_ids = client.get_collection(collection_name).get(include=[])["ids"]
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
        # 1. Yükleme işlemini parçalara (batch) ayırıyoruz
        num_chunks = len(chunks_to_add)

        # for döngüsü ile 0'dan başlayarak, MAX_BATCH_SIZE adımlarıyla ilerle
        for i in range(0, num_chunks, MAX_BATCH_SIZE):

            # Başlangıç ve bitiş indexlerini belirle
            end_index = min(i + MAX_BATCH_SIZE, num_chunks)

            # Chunk ve ID'leri bu batch için ayır
            batch_chunks = chunks_to_add[i:end_index]
            batch_ids = ids_to_add[i:end_index]

            print(
                f"   -> Batch {int(i/MAX_BATCH_SIZE) + 1}: {len(batch_chunks)} parça yükleniyor..."
            )

            try:
                # 2. Batch'i yükle
                vectorstore.add_documents(documents=batch_chunks, ids=batch_ids)

            except Exception as e:
                print(
                    f"!!! YÜKLEME HATASI BATCH {int(i/MAX_BATCH_SIZE) + 1} !!! Hata: {e}"
                )
                # Hata durumunda döngüden çıkılabilir veya hata loglanıp devam edilebilir
                break

    print(f"\n--- Yükleme Özeti ---")
    print(f"Toplam Parça İşlendi: {len(all_chunks)}")
    print(f"✅ Yeni Eklendi: {new_count} adet parça.")
    print(f"🔄 Güncellendi (Overwrite Edildi): {updated_count} adet parça.")
    print("Veritabanı başarıyla güncellendi (Batching Kullanıldı).")

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
    vectorstore = db_update()
