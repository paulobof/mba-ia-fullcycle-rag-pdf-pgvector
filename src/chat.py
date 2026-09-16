"""CLI de perguntas e respostas sobre o conteúdo ingerido do PDF."""

from langchain_core.runnables import Runnable

from search import build_chain

EXIT_COMMANDS = frozenset({"sair", "exit", "quit"})
BANNER = (
    "Chat pronto. Faça sua pergunta sobre o documento "
    f"(digite {'/'.join(sorted(EXIT_COMMANDS))} para encerrar).\n"
)


def start_chain() -> Runnable[str, str] | None:
    """Inicializa a chain de busca tratando falhas de configuração.

    Returns:
        A chain pronta, ou `None` quando a inicialização falha.
    """
    try:
        return build_chain()
    except Exception as error:  # noqa: BLE001 - fronteira da CLI
        print(f"Erro ao inicializar: {error}")
        return None


def answer(chain: Runnable[str, str], question: str) -> str:
    """Executa a chain para uma pergunta.

    Args:
        chain: Chain de RAG já inicializada.
        question: Pergunta do usuário.

    Returns:
        A resposta da LLM ou uma mensagem de erro amigável.
    """
    try:
        return chain.invoke(question)
    except Exception as error:  # noqa: BLE001 - fronteira da CLI
        return f"Erro ao consultar: {error}"


def main() -> None:
    """Executa o loop principal do chat no terminal."""
    chain = start_chain()

    if chain is None:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print(BANNER)
    while True:
        try:
            question = input("PERGUNTA: ").strip()
        except EOFError, KeyboardInterrupt:
            print("\nAté mais!")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Até mais!")
            return

        print(f"RESPOSTA: {answer(chain, question)}\n")


if __name__ == "__main__":
    main()
