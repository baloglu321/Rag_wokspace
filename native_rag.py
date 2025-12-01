from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_classic.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_classic.chains.llm import LLMChain
from croma_db_update import db_update, update_db_with_feedback

CLOUDFLARE_TUNNEL_URL = "..."
OLLAMA_MODEL_ID = "gemma3:27b"
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# 1. Veritabanını Güncelle ve VectorStore'u Al
print("--- 1. Veritabanı Güncelleme ---")
vectorstore = db_update()
TOP_K_RETRIEVAL = 50  # ChromaDB'den çekilecek toplam parça sayısı
TOP_K_RERANK = 10   # LLM'e gönderilecek nihai parça sayısı


def get_reranked_documents(vectorstore: Chroma, query: str):
    """ChromaDB'den çok sayıda parça çeker ve bunları Cross-Encoder ile yeniden sıralar."""
    
    # 1. ChromaDB'den daha fazla parça çekme (k=10)
    retrieved_docs = vectorstore.similarity_search(query, k=TOP_K_RETRIEVAL)
    
    # 2. Yeniden Sıralama için Veri Hazırlama
    # Cross-Encoder, (sorgu, metin) çiftleri listesi ister.
    sentences = [(query, doc.page_content) for doc in retrieved_docs]
    
    # 3. Puanlama (Score)
    # Model, her parça için 0-1 arasında bir alaka puanı hesaplar.
    scores = reranker.predict(sentences)
    
    # 4. Parçaları Puanlarla Birleştirme ve Sıralama
    doc_scores = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    
    # 5. En iyi (TOP_K_RERANK) parçayı seçme
    final_docs = [doc_score[0] for doc_score in doc_scores[:TOP_K_RERANK]]
    
    return final_docs


def create_reranked_rag_chain(vectorstore: Chroma):
    """
    Özelleştirilmiş Yeniden Sıralama (Re-ranking) mantığını kullanarak 
    RAG zincirini oluşturur.
    """
    llm = OllamaLLM(model=OLLAMA_MODEL_ID, base_url=CLOUDFLARE_TUNNEL_URL)

    template = """
    Senin görevin, sadece ve sadece AŞAĞIDAKİ BAĞLAM'da bulunan bilgilere dayanarak yanıt vermektir. 
    BAĞLAM, gerçeğin tek kaynağıdır.

    Eğer BAĞLAM'daki bilgiler, soruyu doğrudan yanıtlamak için yeterli DEĞİLSE, kesinlikle ek bilgi eklemeyip, 
    'BAĞLAM, bu soruyu yanıtlamak için yeterli bilgi sağlamamaktadır.' cümlesini kullan. 

    BAĞLAM:
    {context}

    SORU: {question}

    Yanıt:
    """
    RAG_PROMPT = PromptTemplate(
        template=template, 
        input_variables=["context", "question"]
    )

    # RetrievalQA zincirini, LLM'e doğrudan Parçaları besleyen 'stuff' tipiyle oluşturuyoruz.
    # Ancak retriever'ı, yeniden sıralama mantığımızı uygulayacak özel bir fonksiyondan almalıyız.
    # LangChain'in RetrievalQA'sı esnek olmadığı için, sorgu mantığını manuel olarak uygulamamız en temiz yol olacaktır.
    
    # Bu adımı basitleştirmek adına, ana çalışma bloğunda manuel Retrieval/Reranking yapıp, 
    # ardından LLM'i çağıracak bir fonksiyon oluşturalım.
    
    # RetrievalQA yerine, sadece LLM'in kendisini döndürelim ve ana döngüde manuel olarak kullanalım:
    return llm, RAG_PROMPT

def test_reranked_rag_query(llm, prompt, vectorstore, query: str):
    """Yeniden sıralama mantığını uygulayarak sorguyu çalıştırır ve sonuçları yazdırır."""
    
    print(f"\nSorgu: {query}")
    print("-" * 50)
    
    # 1. Yeniden Sıralamayı Uygula (Retrieval + Reranking)
    final_docs = get_reranked_documents(vectorstore, query)
    
    # 2. Bağlamı Hazırla
    context = "\n---\n".join([doc.page_content for doc in final_docs])
    
    # 3. LLM Chain'i Oluştur ve Çalıştır
    LLMChain(prompt=prompt, llm=llm)
    
    # Final Prompt'u oluştur
    final_prompt = prompt.format(context=context, question=query)
    
    # Cevabı al
    response = llm.invoke(final_prompt)
    
    # Cevabı Yazdırma
    print(f"\n✅ Gemma Cevabı: {response}")
    print("\n--- Kullanılan Kaynaklar (Yeniden Sıralanmış TOP 3 Parça) ---")
    
    for doc in final_docs:
        print(f"Kaynak Dosya: {doc.metadata.get('source', 'Bilinmiyor')}")
        print(f"İçerik Başlangıcı: {doc.page_content[:150]}...")

if __name__ == "__main__":
    
    # 1. Veritabanını Güncelle ve VectorStore'u Al
    print("--- 1. Veritabanı Güncelleme ---")
    vectorstore = db_update() # Bu fonksiyonun başarıyla çalıştığını varsayıyoruz
    
    # 2. LLM ve Prompt'u Al
    print("\n--- 2. RAG Bileşenleri Kurulumu ---")
    llm_model, rag_prompt = create_reranked_rag_chain(vectorstore)
    print("Yeniden Sıralama (Re-ranking) Modeli ve LLM hazır.")
    
    # 3. Zorlu Sorguları Çalıştır
    
    # Eski Sorgu: Arif'in, Ceku'nun gerçek babasının kim olduğunu öğrenmesinden sonra Gora'ya dönmek için kimin gemisini kullandığına dair bir kanıt var mı?
    test_reranked_rag_query(
        llm_model, rag_prompt, vectorstore, 
        query="Ceku karakterinin Babası kimdir?"
    )

    # Eski Sorgu: Gora'daki 'alev topu' krizinin aslında bir aldatmaca olduğunu gösteren ve bu aldatmacayı hazırlayan kişinin planına dair LLM'in bulduğu detaylar nelerdir?
    test_reranked_rag_query(
        llm_model, rag_prompt, vectorstore, 
        query="Gora'daki 'alev topu' krizinin sorumlusu kimdir? Alev topunu goraya kim göndermiştir?"
    )
    
    # Yeni Sorgu: AROG filminde, Arif'in medeniyeti kurmasına karşı çıkan köy liderinin adı neydi ve icatlara karşı çıkmasının nedeni neydi?
    test_reranked_rag_query(
        llm_model, rag_prompt, vectorstore, 
        query="AROG filminde, Karga karakterinin babası kimdir?"
    )
