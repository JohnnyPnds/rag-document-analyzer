#Import das Bibliotecas utilizadas
from pypdf import PdfReader #Importação de PDF para o Python
from langchain_text_splitters import RecursiveCharacterTextSplitter #Responsável pela quebra do texto em fatias
from langchain_huggingface import HuggingFaceEmbeddings #Responsável por entender o contexto ao invés de localizar pela palavra exata
from langchain_chroma import Chroma #armazena textos em coordenadas matemáticas
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq



#Criando função responsável por contar as páginas do PDF
def extrair_texto_pdf(caminho_do_arquivo):
    leitor = PdfReader(caminho_do_arquivo)
    texto_extraido= ""

    #looping de leitura de uma pagina por vez
    for pagina in leitor.pages:
        texto_extraido += pagina.extract_text() + "\n" #\n utilizado para pular linha no final das páginas

    return texto_extraido

arquivo_pdf = "harvard_ai_review.pdf" #define o arquivo de entrada
texto_real = extrair_texto_pdf(arquivo_pdf) #chama uma função que le o pdf e retorna uma string única

#Definindo métricas do fatiador
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

#utilizando o splitter com o texto extraído do PDF
fatias = splitter.split_text(texto_real)

#Configurando o Banco Vetorial
print("Conectando ao banco de dados vetorial...")
modelo_nome = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name = modelo_nome)
pasta_db = "meu_banco_vetorial"



#verificação se o banco já existe ou não
if not os.path.exists(pasta_db):
    #banco será criado do zero
    vector_store = Chroma.from_texts(
    texts = fatias,
    embedding = embeddings,
    persist_directory = pasta_db
    )
    print("Banco criado com sucesso!")
else:
    #banco já existe, então não será necessário ler o pdf novamente
    print("Banco existente localizado. Carregando...")
    vector_store = Chroma(
        persist_directory = pasta_db,
        embedding_function = embeddings
    )

#Pergunta do usuário
pergunta = "Segundo o documento de Harvard, qual a importância da Engenharia de Software para a Ciência de Dados?"
print(f"/n consultando o banco: {pergunta}")

#Buscando os k trechos mais semelhantes
docs = vector_store.similarity_search(pergunta, k=3)

load_dotenv()

# Buscamos a chave que foi carregada do .env
chave_api = os.getenv("GROQ_API_KEY")

# Inicializamos o LLM passando a chave diretamente
llm = ChatGroq(
    model_name = "llama-3.1-8b-instant",
    groq_api_key=chave_api
)

#unificando os trechos selecionados em um só para passar como contexto
contexto = "\n\n".join([doc.page_content for doc in docs])

prompt_sistema = f"""
Você é um assistente acadêmico especialista.
Abaixo estão trechos de um documento de Harvard em inglês.
Sua tarefa é responder à pergunta do usuário em PORTUGUÊS BRASILEIRO usando APENAS o contexto fornecido.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}
"""

print(f"\n ---SOLICITANDO RESPOSTA A GROQ---")
#enviando prompt para a ia
resposta_final = llm.invoke(prompt_sistema)

#exibindo resposta da ia
print("\n ===>RESPOSTA FINAL DA IA<===")
print(resposta_final.content)