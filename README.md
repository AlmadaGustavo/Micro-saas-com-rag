# Micro-saas-com-rag
# EditalIA — Análise Inteligente de Editais Públicos

Sistema de perguntas e respostas sobre editais de licitação, concursos públicos e vestibulares, baseado em RAG (Retrieval-Augmented Generation) com modelos de linguagem locais via Ollama.

---

## Equipe
Gustavo Morais de Almada             
Rafael Santos             
Leonardo Melo Crespim          

---

## Descrição do Projeto

O EditalIA permite que o usuário faça upload de um PDF de edital e faça perguntas em linguagem natural sobre o documento. O sistema recupera os trechos mais relevantes e gera uma resposta fundamentada, citando a página e a cláusula de origem.

Exemplos de perguntas suportadas:

- Qual o prazo para entrega das propostas?
- Quais documentos são obrigatórios para habilitação?
- Quais são os critérios de desclassificação?
- Qual o objeto desta licitação?

---

## Arquitetura

```
PDF do edital
     │
     ▼
pdf_extractor.py   — Extrai blocos de texto com metadados tipográficos (PyMuPDF)
     │
     ▼
chunker.py         — Gera pares Parent-Child via chunking hierárquico
     │
     ▼
indexer.py         — Persiste no ChromaDB com embeddings via Ollama
     │
     ▼
retriever.py       — Busca híbrida + expansão de contexto + geração via LLM
     │
     ▼
app.py             — Interface Streamlit
```

### Estratégia de Chunking

O sistema usa **Parent-Child chunking** em dois níveis:

- **Parent (~700 tokens):** contexto completo de uma cláusula. Enviado ao LLM para geração da resposta.
- **Child (~175 tokens):** fragmento menor do mesmo trecho. Indexado por embedding para busca semântica precisa.

Na recuperação, o sistema busca pelos children (semântica fina) e expande para os parents (contexto completo) antes de montar o prompt.

### Metadados por Chunk

Cada chunk carrega metadados filtráveis no ChromaDB:

| Metadado | Descrição |
|---|---|
| `chunk_type` | Classificação semântica: `prazo`, `documento_obrigatorio`, `criterio_desclassificacao`, `objeto`, `outro` |
| `section_title` | Título da seção pai (ex: "3. DA HABILITAÇÃO") |
| `clause_id` | Número da cláusula (ex: "3.1.2") |
| `page_start` | Página do PDF onde o trecho começa |
| `contains_date` | `1` se o chunk contém uma data no formato DD/MM/AAAA |
| `contains_value` | `1` se o chunk contém um valor monetário (R$) |
| `edital_id` | SHA-256 (8 chars) do arquivo — permite múltiplos editais no mesmo banco |

---

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem | Python 3.10+ |
| Extração de PDF | PyMuPDF (fitz) |
| Banco vetorial | ChromaDB |
| Embeddings | nomic-embed-text via Ollama |
| LLM | Llama 3 / Mistral via Ollama |
| Orquestração | LangChain Core |
| Interface | Streamlit |

---

## Pré-requisitos

- Python 3.10 ou superior
- [Ollama](https://ollama.com/download) instalado e rodando

---

## Instalação

**1. Clonar o repositório:**

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

**2. Criar ambiente virtual e instalar dependências:**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

**3. Baixar os modelos no Ollama:**

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

---

## Execução

**1. Iniciar o Ollama** (se não estiver rodando):

```bash
ollama serve
```

**2. Iniciar a interface:**

```bash
streamlit run app.py
```

**3. Usar o sistema:**

1. Faça upload de um PDF de edital no painel lateral
2. Clique em "Processar e Indexar" e aguarde a indexação
3. Digite sua pergunta na caixa de chat

---

## Estrutura do Projeto

```
├── app.py                    # Interface Streamlit
├── retriever.py              # Pipeline RAG (busca + geração)
├── requirements.txt
├── ingestion/
│   ├── __init__.py
│   ├── pdf_extractor.py      # Extração estrutural de PDFs
│   ├── chunker.py            # Chunking hierárquico Parent-Child
│   └── indexer.py            # Indexação e busca no ChromaDB
└── chroma_db/                # Banco vetorial (gerado em runtime)
```

---

## Pipeline de Ingestão — Detalhes Técnicos

### 1. Extração (`pdf_extractor.py`)

O extrator usa `get_text("dict")` do PyMuPDF para obter cada span com seus atributos tipográficos (tamanho de fonte, negrito, posição). A partir disso:

- Reconstrói a hierarquia do documento (Seção > Cláusula > Parágrafo) sem depender de tags HTML
- Detecta o tamanho de fonte do corpo por frequência ponderada (funciona para qualquer órgão emitente)
- Classifica cada bloco semanticamente por contagem de padrões regex
- Extrai metadados de entidades: datas, valores monetários, CNPJs, referências cruzadas entre cláusulas

### 2. Chunking (`chunker.py`)

- Agrupa blocos em seções lógicas (cada heading inicia uma nova seção)
- Gera Parents de até 700 tokens com overlap de 100 tokens entre janelas
- Subdivide cada Parent em Children de até 175 tokens com overlap de 30 tokens
- Propaga todos os metadados da seção para parents e children

### 3. Indexação (`indexer.py`)

- Persiste children na coleção `editais_children` com embedding real (busca semântica)
- Persiste parents na coleção `editais_parents` com embedding dummy (recuperação por ID)
- Operação idempotente: IDs já existentes são ignorados (deduplicação por SHA-256)
- Suporta múltiplos editais no mesmo banco via filtro por `edital_id`

### 4. Recuperação (`retriever.py`)

- Detecta a intenção da pergunta (prazo, documento, critério, objeto) para pré-filtrar a busca
- Busca híbrida em duas camadas: filtrada por chunk_type + semântica pura, mescladas por parent_id
- Expande cada child recuperado para seu parent (contexto completo)
- Monta prompt com trechos numerados e cabeçalhos de localização (seção, cláusula, página)
- Temperatura 0.0 no LLM para respostas determinísticas em documentos jurídicos

---

## Decisões de Design

**Por que duas coleções no ChromaDB?**  
Separar children (busca) de parents (contexto) permite que a busca semântica opere em textos curtos e precisos, enquanto o LLM recebe o contexto completo da cláusula. Uma coleção única misturaria os dois e prejudicaria a qualidade da busca.

**Por que nomic-embed-text em vez do modelo padrão do ChromaDB?**  
O modelo padrão (all-MiniLM-L6-v2) é treinado principalmente em inglês. O nomic-embed-text é multilingual e produz distâncias coseno de 0.20–0.30 para textos relevantes em português, contra 0.40–0.62 do modelo padrão.

**Por que temperatura 0.0 no LLM?**  
Editais têm linguagem jurídica precisa. Um LLM com temperatura alta pode parafrasear prazos ou valores de forma incorreta. Temperatura zero garante que o modelo copia fielmente as informações do contexto.

**Por que chunking por palavras e não por tokens exatos?**  
A heurística `palavras × 1.35` aproxima bem o comportamento de tokenizadores para português (±5%) sem exigir dependência de tiktoken, que requer acesso à internet para download do vocabulário.

---

## Limitações Conhecidas

- PDFs escaneados (sem texto nativo) têm qualidade de extração reduzida. Recomenda-se pré-processar com OCR antes do upload.
- O classificador de `chunk_type` usa regex e pode errar em editais com linguagem muito atípica. É possível retreinar os padrões editando `_PATTERNS` no `pdf_extractor.py`.
- O tempo de indexação depende da velocidade do Ollama local.