# Desafio MBA IA — Ingestão e Busca Semântica

RAG em Python que ingere um PDF no PostgreSQL com pgVector e responde perguntas no terminal
usando **apenas** o conteúdo do documento.

- **Ingestão:** PDF → chunks de 1000 caracteres com overlap de 150 → embeddings → pgVector.
- **Busca:** pergunta → embedding → `similarity_search_with_score(query, k=10)` → corte por
  relevância → prompt → LLM.
- **Fora de contexto:** responde sempre `Não tenho informações necessárias para responder sua pergunta.`

### Como a recusa é garantida

O corte por relevância (`MIN_RELEVANCE = 0.35`) é o que torna a recusa estrutural em vez de
depender da obediência do modelo: quando nenhum trecho atinge o limiar, o contexto chega
vazio à LLM. Medido no documento do desafio:

| Pergunta | Trechos recuperados | Trechos usados |
|---|---|---|
| Dentro do documento | 10 | 10 |
| Fora do documento | 10 | **0** |

Reforçam o grounding: regras numa mensagem `system` separada, contexto delimitado por
`<contexto>` (tratado como dado, nunca como instrução) e a regra de recusa repetida
**depois** do contexto, para não ficar soterrada pelos trechos recuperados.

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.14 (fixado em `.python-version`) |
| Gerenciador | [uv](https://docs.astral.sh/uv/) |
| Framework | LangChain 1.x |
| Banco vetorial | PostgreSQL 17 + pgVector (Docker) |
| Configuração | Pydantic Settings v2 |
| Qualidade | pytest + coverage, mypy (strict), ruff |
| Providers | OpenAI **ou** Google Gemini (escolha via `.env`) |

## Pré-requisitos

- Docker e Docker Compose
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Uma API key: OpenAI **ou** Google Gemini

## Configuração

```bash
cp .env.example .env   # preencha a API key do provider escolhido
make setup             # uv sync --all-groups (cria .venv e instala tudo)
```

Variáveis do `.env`:

| Variável | Padrão | Descrição |
|---|---|---|
| `PROVIDER` | `openai` | `openai` ou `gemini` |
| `OPENAI_API_KEY` | — | Obrigatória quando `PROVIDER=openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 1536 dimensões |
| `OPENAI_LLM_MODEL` | `gpt-5.6-luna` | Modelo de resposta (família gpt-5 roda com `reasoning_effort=none` para preservar `temperature=0`) |
| `GOOGLE_API_KEY` | — | Obrigatória quando `PROVIDER=gemini` |
| `GOOGLE_EMBEDDING_MODEL` | `gemini-embedding-001` | 3072 dimensões |
| `GOOGLE_LLM_MODEL` | `gemini-2.5-flash-lite` | Modelo de resposta |
| `DATABASE_URL` | `postgresql+psycopg://postgres:postgres@localhost:5432/rag` | Conexão SQLAlchemy |
| `PG_VECTOR_COLLECTION_NAME` | `desafio_rag` | Nome da collection |
| `PDF_PATH` | `document.pdf` | PDF a ser ingerido |

A configuração é validada no start: se faltar a chave do provider escolhido, a aplicação falha
com mensagem explícita em vez de estourar na primeira chamada de API.

## Ordem de execução

```bash
# 1. Subir o banco (Postgres 17 + extensão vector)
docker compose up -d        # ou: make db-up

# 2. Ingerir o PDF
uv run python src/ingest.py # ou: make ingest

# 3. Conversar com o documento
uv run python src/chat.py   # ou: make chat
```

Exemplo de sessão:

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Digite `sair`, `exit` ou `quit` (ou `Ctrl+C`) para encerrar.

## Sem uv?

O `requirements.txt` é gerado a partir do `uv.lock` (`make requirements`) e continua válido —
exige Python 3.14, versão fixada em `.python-version`:

```bash
python3.14 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py
python src/chat.py
```

## Estrutura

```
├── docker-compose.yml     # Postgres 17 + pgVector + bootstrap da extensão
├── document.pdf           # documento ingerido
├── pyproject.toml         # dependências (uv), pytest, mypy e ruff
├── requirements.txt       # export do uv.lock, para quem não usa uv
├── Makefile               # atalhos: setup, ingest, chat, check
├── src/
│   ├── config.py          # Settings (Pydantic v2) validadas
│   ├── providers.py       # fábrica de embeddings, LLM e vector store
│   ├── ingest.py          # carga do PDF, split 1000/150 e gravação
│   ├── search.py          # busca k=10, corte por relevância, prompt e chain
│   └── chat.py            # CLI
└── tests/                 # pytest com dublês (sem rede)
```

## Qualidade

```bash
make check   # ruff + mypy strict + pytest com cobertura
```

- Cobertura atual: **99%** (41 testes, nenhum chama API externa).
- `mypy` em modo `strict`, `ruff` com regras de docstring, tipagem e bugs comuns.

## Decisões de implementação

- **Ids determinísticos na ingestão:** cada chunk recebe um `sha256` de origem, página, posição
  e conteúdo. Reexecutar `ingest.py` faz *upsert*, não duplica vetores.
- **`PGVector` (deprecado) em vez de `PGVectorStore`:** é a classe pedida no enunciado do
  desafio; segue exportada pelo `langchain-postgres` 0.0.18.
- **Chave de API tipada como `SecretStr`:** não aparece em logs nem em `repr()`.
- **`temperature=0`:** aplicada quando o modelo aceita. Modelos de raciocínio (família `gpt-5.x`)
  ignoram o parâmetro por design da API.
- **Python 3.14 obrigatório:** o projeto usa sintaxe da versão (PEP 758, `except A, B:`) e
  anotações adiadas (PEP 649). `ruff`, `mypy` e `uv` estão todos fixados em `py314`.

## Troubleshooting

| Problema | Causa | Solução |
|---|---|---|
| `expected N dimensions, not M` | Troca do modelo de embeddings após a primeira ingestão | `docker compose down -v && docker compose up -d` e reingerir |
| `OPENAI_API_KEY é obrigatória...` | `.env` sem a chave do provider ativo | Preencher a chave ou trocar `PROVIDER` |
| `PDF não encontrado` | `PDF_PATH` inválido | Corrigir o caminho no `.env` |
| `connection refused` na porta 5432 | Banco não subiu | `docker compose ps` e `docker compose up -d` |
