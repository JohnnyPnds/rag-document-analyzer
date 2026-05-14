RAG DOCUMENT ANALYZER

Este projeto implementa um sistema de Retrieval-Augmented Generation (RAG) que permite consultar documentos técnicos complexos em inglês e obter respostas inteligentes e contextualizadas em português brasileiro.

O sistema utiliza o editorial "Data Science and Engineering With Human in the Loop" da Harvard Data Science Review como base de conhecimento e exemplo de aplicação.

🧠 Arquitetura do Projeto
O software segue o fluxo clássico de uma pipeline de dados para IA:

Ingestão de Dados: Extração de texto bruto de arquivos PDF utilizando PyPDF.

Processamento (Chunking): Divisão do texto em fatias menores (RecursiveCharacterTextSplitter) para respeitar a janela de contexto do modelo.

Vetorização (Embeddings): Conversão de texto em vetores matemáticos usando o modelo all-MiniLM-L6-v2 da HuggingFace.

Armazenamento Vetorial: Persistência dos dados no ChromaDB, permitindo que o sistema "lembre" do documento sem precisar reprocessá-lo a cada execução.

Recuperação (Retrieval): Busca por similaridade de cosseno para encontrar os trechos mais relevantes para a pergunta do usuário.

Geração: Uso da API da Groq (Llama 3.1) para sintetizar a resposta final de forma clara e traduzida.

🛠️ Tecnologias Utilizadas
Linguagem: Python 3.11+

Orquestração de IA: LangChain

Banco de Dados: ChromaDB

Modelo de Linguagem: Llama 3.1 8B via Groq

Embeddings: HuggingFace

⚙️ Configuração e Instalação
1. Requisitos
Python instalado.

Uma API Key da Groq Console.

2. Variáveis de Ambiente
Crie um arquivo .env na raiz do projeto para armazenar suas credenciais de forma segura:

Plaintext
GROQ_API_KEY=sua_chave_aqui
3. Instalação de Dependências

pip install pypdf langchain-text-splitters langchain-huggingface langchain-chroma langchain-groq python-dotenv
🚀 Como usar
Basta executar o script principal:

python analyze_document.py
O sistema irá identificar o banco de dados existente ou criará um novo caso seja a primeira execução, processando o documento e respondendo à pergunta pré-configurada ou inserida pelo usuário.