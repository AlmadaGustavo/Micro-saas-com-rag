# EditalIA — Análise Inteligente de Editais Públicos com Agentic RAG

O **EditalIA** é um sistema de auditoria, perguntas e respostas sobre editais de licitação, concursos públicos e vestibulares. O projeto utiliza uma arquitetura avançada de **Agentic RAG (Retrieval-Augmented Generation)** controlada por um grafo de estados para implementar loops autônomos de avaliação e auto-correção (*Self-RAG*), utilizando modelos de linguagem e embeddings locais via Ollama.

---

## 👥 Equipe
* **Gustavo Morais de Almada** * **Rafael Santos** * **Gabriel Pepes Moda**

---

## 📝 Descrição do Projeto

O **EditalIA** permite que analistas de mercado e inteligência de negócios façam upload de editais públicos em formato PDF e realizem consultas em linguagem natural. 

Diferente de sistemas RAG tradicionais que executam buscas lineares e estáticas, o EditalIA atua como um **agente autônomo de tomada de decisão**. Ele analisa a qualidade dos documentos recuperados, valida a relevância semântica em relação à dúvida do usuário e, caso a busca inicial falhe ou seja ambígua, reescreve a pergunta de forma técnica para forçar uma nova varredura resiliente no banco vetorial antes de sintetizar a resposta final com rastreabilidade de fontes.

---

## 🏗️ Arquitetura do Sistema (Fluxo Agêntico)

Abaixo está o mapeamento da máquina de estados finitos que controla o ciclo de vida de uma consulta no backend do sistema:

[ Pergunta do Usuário ]
                         │
                         ▼
              ┌────────────────────┐
              │  route_and_search  │ ◄───────────────────┐
              │  (Busca Híbrida)   │                     │
              └──────────┬─────────┘                     │
                         │                               │
                         ▼                               │
              ┌────────────────────┐                     │
              │  grade_documents   │                     │
              │ (Agente Crítico)   │                     │
              └──────────┬─────────┘                     │
                         │                               │
                         ├───────────────────────────────┤ (Se a busca falhar
                         │ Condicional:                  │  e loop_count < 2)
                         ▼ decide_to_generate            │
               [ relevant_chunks? ]                      │
                 ├── SIM ────────────────► [ generate_answer ]
                 │                             (Geração Final)
                 └── NÃO ────────────────► [ rewrite_query ]
                                               (Agente Tradutor)
### Componentes e Módulos do Sistema

* **`ingestion/pdf_extractor.py`**: Camada de extração estrutural. Avalia a tipografia do PDF através do PyMuPDF, mapeia a hierarquia física e realiza a classificação semântica inicial.
* **`ingestion/chunker.py`**: Camada de processamento hierárquico. Implementa a estratégia *Parent-Child* e aplica algoritmos de correção de fragmentos órfãos em listas e resoluções.
* **`ingestion/indexer.py`**: Interface de persistência com o ChromaDB. Gerencia coleções isoladas para os pares hierárquicos e expõe métodos para busca híbrida e recuperação por ID exato (`fetch_by_clause_id`).
* **`agent.py`**: O cérebro do ecossistema. Implementa a máquina de estados através do **LangGraph**, gerenciando os nós de execução, avaliação crítica, loops de auto-correção e as bordas de roteamento condicional.
* **`app.py`**: Camada de interface de usuário construída em Streamlit, alimentada pelas respostas estruturadas do grafo.

---

## 🗂️ Estratégia Avançada de Dados

### 1. Chunking Hierárquico Parent-Child
Para garantir que a busca vetorial opere com alta precisão sem que o LLM perca o contexto periférico do documento, o sistema cinde os dados em duas coleções distintas no ChromaDB:
* **Parents (`editais_parents`):** Blocos robustos de até **700 tokens** com a cláusula ou seção jurídica completa. Não geram vetores de embedding (armazenados com vetor dummy `[0.0]`) e são injetados diretamente no prompt do LLM.
* **Children (`editais_children`):** Subdivisões do bloco pai com até **175 tokens**, indexadas com embeddings reais para máxima aderência matemática na busca semântica. Cada registro filho carrega o ID do seu respectivo pai.

### 2. Metadados e Suporte a Multi-Hop (Referências Cruzadas)
Cada bloco vetorial armazena metadados primitivos para pré-filtragem acelerada e rastreabilidade:

| Metadado | Tipo | Função no Sistema |
|---|---|---|
| `chunk_type` | `str` | Classificação do tema: `prazo`, `documento_obrigatorio`, `criterio_desclassificacao`, `objeto`, `cabecalho`, `outro` |
| `clause_id` | `str` | Identificador numérico exato da cláusula (Ex: `5.1.2`) |
| `page_start` | `int` | Número da página de origem no PDF para citação compulsória |
| `contains_date` | `int` | Flag booleano (`1` ou `0`) para identificar presença de cronogramas |
| `contains_value` | `int` | Flag booleano (`1` ou `0`) para identificar dotações orçamentárias |
| `referenced_clauses`| `str` | Lista de referências cruzadas mapeadas por regex (Ex: `4.1.1,8.2`) salvas como string separada por vírgula para permitir buscas em cascata (*Multi-Hop Retrieval*). |
| `edital_id` | `str` | Hash SHA-256 estável do arquivo, garantindo isolamento multi-tenant de dados |

---

## 🛠️ O Loop de Decisão Agêntica (Self-RAG)

O pipeline implementado no módulo `agent.py` elimina o comportamento estático do RAG clássico através de 4 nós cognitivos:

1. **Roteamento Temático e Busca (`node_route_and_search`):** Executa o algoritmo de intenção na pergunta. Se uma palavra-chave ativar um padrão, o grafo injeta um filtro `where` nativo no ChromaDB para buscar apenas na subcategoria correspondente, reduzindo o espaço de busca. Em seguida, os filhos encontrados são expandidos para seus blocos pais originais.
2. **Auditoria Crítica (`node_grade_documents`):** O LLM é instanciado em modo de avaliação restrita (temperatura `0.0`). Ele lê a pergunta do usuário e cada um dos blocos retornados individualmente, emitindo um veredito binário (`SIM` ou `NAO`) se aquele trecho possui utilidade factual para a resposta.
3. **Avaliação Estatística de Estado (`decide_to_generate`):** Uma borda condicional avalia o estado do grafo. Se houver blocos aprovados, o fluxo avança para a síntese. Se nenhum bloco for aprovado, o estado incrementa o `loop_count` e desvia o fluxo para a agência de correção.
4. **Reformulação Cognitiva da Query (`node_rewrite_query`):** O LLM atua como um tradutor técnico de mercado. Ele analisa a pergunta que falhou e a reescreve utilizando sinônimos formais de compras públicas e jargões da Lei 14.133 para otimizar a aproximação dos vetores em uma nova tentativa de busca automática.

---

## 💻 Stack Tecnológica

* **Linguagem:** Python 3.10+
* **Orquestração Agêntica:** LangGraph & LangChain Core
* **Interface Gráfica:** Streamlit
* **Banco de Vetores:** ChromaDB
* **Extração de PDF e Tipografia:** PyMuPDF (fitz)
* **Modelos Locais (Ollama):**
  * *LLM:* Llama 3 (8B) / Mistral (7B) com temperatura `0.0`
  * *Embeddings:* `nomic-embed-text` (Modelo multilíngue de alta performance para a língua portuguesa)

---

## 🚀 Instalação e Execução

**1. Clonar o repositório e acessar a pasta:**
```bash
git clone <url-do-repositorio>
cd Micro-saas-com-rag

python -m venv .venv

# Ativação no Windows
.venv\Scripts\activate

# Ativação no Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt

ollama serve
ollama pull llama3
ollama pull nomic-embed-text
````

## Execução do projeto

Com o Ollama em execução
**streamlit run app.py**

##Estrutura de arquivos
├── app.py                    # Interface Streamlit customizada
├── agent.py                  # Orquestração do Grafo LangGraph e Dataclasses RAG
├── requirements.txt          # Dependências do ecossistema
├── ingestion/
│   ├── __init__.py
│   ├── pdf_extractor.py      # Motor tipográfico e extrator de spans (PyMuPDF)
│   ├── chunker.py            # Algoritmo de fatiamento hierárquico e janelas de overlap
│   └── indexer.py            # Gerenciador idempotente de coleções (ChromaDB)
└── chroma_db/                # Diretório de persistência do banco em runtime
