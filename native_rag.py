from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
import chromadb
from langchain_classic.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from langchain_classic.chains.llm import LLMChain
from langchain_huggingface import HuggingFaceEmbeddings


CLOUDFLARE_TUNNEL_URL = (
    "https://subscriptions-elephant-leaving-appearing.trycloudflare.com/"
)
OLLAMA_MODEL_ID = "gemma3:27b"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2"
CHROMA_HOST = "localhost"  # Sadece ana bilgisayar adı
CHROMA_PORT = 8000
COLLECTION_NAME = "rag_test_data"
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# 1. Veritabanını Güncelle ve VectorStore'u Al
print("--- 1. Veritabanı Güncelleme ---")

TOP_K_RETRIEVAL = 30  # ChromaDB'den çekilecek toplam parça sayısı
TOP_K_RERANK = 10  # LLM'e gönderilecek nihai parça sayısı


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
    You are an expert Question-Answering system. Your sole instruction is to generate a final answer 
    BASED ONLY on the information provided in the CONTEXT below. The CONTEXT is your only source of truth.

    You MUST answer the question in the SAME LANGUAGE as the question was asked. 
    (E.g., if the question is in Turkish, the answer MUST be in Turkish.)

    If the CONTEXT is insufficient to fully answer the question, you MUST NOT add external information. 
    Instead, you must summarize the most relevant information you found in the context and end your response 
    with a clarification that the context was not sufficient.

    If the CONTEXT contains absolutely NO relevant information, respond with: 
    "I am sorry, this topic is not covered in the provided context."

    CONTEXT:
    {context}

    QUESTION: {question}

    Answer:
    """
    RAG_PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
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
    """print("\n--- Kullanılan Kaynaklar (Yeniden Sıralanmış TOP 3 Parça) ---")
    
    for doc in final_docs:
        print(f"Kaynak Dosya: {doc.metadata.get('source', 'Bilinmiyor')}")
        print(f"İçerik Başlangıcı: {doc.page_content[:150]}...")
"""


if __name__ == "__main__":
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    embedding_function = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-mpnet-base-v2"
    )

    # 3. Var olan koleksiyona bağlanın (Yükleme yapmadan)
    vectorstore = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    print("Mevcut veritabanı koleksiyonuna başarıyla bağlanıldı.")
    # 2. LLM ve Prompt'u Al
    print("\n--- 2. RAG Bileşenleri Kurulumu ---")
    llm_model, rag_prompt = create_reranked_rag_chain(vectorstore)
    print("Yeniden Sıralama (Re-ranking) Modeli ve LLM hazır.")

    # 3. Zorlu Sorguları Çalıştır

    # 1. Türkçe Factual Sorgu (Bu zaten Türkçe kaldı)
    test_reranked_rag_query(
        llm_model,
        rag_prompt,
        vectorstore,
        query="Rollo'nun Vikinglerinin torunları hangi dili ve dini benimsedi?",
    )

    # 2. İNGİLİZCE Multi-Hop Sorgu (KUvvet Birimleri)
    # (Önceki: The Newton'dan daha az kullanılan metrik terim nedir ve bazen ne olarak adlandırılır?)
    test_reranked_rag_query(
        llm_model,
        rag_prompt,
        vectorstore,
        query="What is the metric term less used than the Newton, and what is it sometimes referred to?",
    )

    # 3. İNGİLİZCE Factual/Detay Sorgu (Notre Dame)
    # (Önceki: University of Notre Dame'deki Grotto'nun bir kopyası olduğu, Virgin Mary'nin 1858'de göründüğü iddia edilen yer neresidir?)
    test_reranked_rag_query(
        llm_model,
        rag_prompt,
        vectorstore,
        query="What is the location of the grotto that the University of Notre Dame's grotto is a replica of, where the Virgin Mary allegedly appeared in 1858?",
    )

    # 4. Türkçe Çıkarım/Neden Sorgusu (Bu zaten Türkçe kaldı)
    test_reranked_rag_query(
        llm_model,
        rag_prompt,
        vectorstore,
        query="Normanların eski İskandinav dinini ve dilini bırakıp, yerel halkın dinini ve dilini benimsemesindeki temel kültürel adaptasyon süreci nasıldı?",
    )
    test_reranked_rag_query(
        llm_model,
        rag_prompt,
        vectorstore,
        query="Sürtünme gibi muhafazakar olmayan kuvvetler, neden aslında mikroskobik potansiyellerin sonuçları olarak kabul edilir?",
    )
    # 5. İNGİLİZCE Multi-Hop/Detay Sorgu (Super Bowl)
    # (Önceki: Super Bowl 50'de, geleneksel Romen rakamları (L) neden kullanılmamıştır?)
    test_reranked_rag_query(
        llm_model,
        rag_prompt,
        vectorstore,
        query="Why were the traditional Roman numerals (L) not used for Super Bowl 50?",
    )
