"""Testes da CLI de chat."""

from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

import chat

ANSWER = "O faturamento foi de 10 milhões de reais."


def scripted_input(answers: list[str]) -> Any:
    """Cria um `input` falso que devolve as entradas na ordem informada."""
    pending: Iterator[str] = iter(answers)

    def fake_input(_prompt: str = "") -> str:
        try:
            return next(pending)
        except StopIteration as error:
            raise EOFError from error

    return fake_input


def stub_chain(monkeypatch: pytest.MonkeyPatch, response: str = ANSWER) -> list[str]:
    """Substitui a chain por um runnable que registra as perguntas recebidas."""
    asked: list[str] = []

    def respond(question: str) -> str:
        asked.append(question)
        return response

    monkeypatch.setattr(chat, "build_chain", lambda: RunnableLambda(respond))
    return asked


def test_should_print_answer_for_each_question(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked = stub_chain(monkeypatch)
    monkeypatch.setattr("builtins.input", scripted_input(["Qual o faturamento?", "sair"]))

    chat.main()

    assert asked == ["Qual o faturamento?"]
    assert f"RESPOSTA: {ANSWER}" in capsys.readouterr().out


def test_should_ignore_empty_questions(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked = stub_chain(monkeypatch)
    monkeypatch.setattr("builtins.input", scripted_input(["", "   ", "sair"]))

    chat.main()

    assert asked == []
    assert "RESPOSTA:" not in capsys.readouterr().out


@pytest.mark.parametrize("command", ["sair", "EXIT", "Quit", "  sair  "])
def test_should_exit_on_exit_commands_regardless_of_case_and_spacing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    asked = stub_chain(monkeypatch)
    monkeypatch.setattr("builtins.input", scripted_input([command, "sair"]))

    chat.main()

    assert asked == []
    assert "Até mais!" in capsys.readouterr().out


def test_should_exit_gracefully_on_end_of_input(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stub_chain(monkeypatch)
    monkeypatch.setattr("builtins.input", scripted_input([]))

    chat.main()

    assert "Até mais!" in capsys.readouterr().out


def test_should_report_initialization_failure_without_stacktrace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode() -> None:
        raise RuntimeError("DATABASE_URL ausente")

    monkeypatch.setattr(chat, "build_chain", explode)

    chat.main()

    output = capsys.readouterr().out
    assert "DATABASE_URL ausente" in output
    assert "Não foi possível iniciar o chat." in output


def test_should_keep_chat_alive_when_a_query_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def explode(_question: str) -> str:
        raise RuntimeError("timeout na LLM")

    monkeypatch.setattr(chat, "build_chain", lambda: RunnableLambda(explode))
    monkeypatch.setattr("builtins.input", scripted_input(["pergunta", "sair"]))

    chat.main()

    output = capsys.readouterr().out
    assert "Erro ao consultar: timeout na LLM" in output
    assert "Até mais!" in output


def test_should_report_friendly_message_when_vector_store_connection_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class OperationalError(Exception):
        """Simula a falha de conexão levantada pelo driver do Postgres."""

    def explode(_question: str) -> str:
        raise OperationalError("could not connect to server: Connection refused")

    monkeypatch.setattr(chat, "build_chain", lambda: RunnableLambda(explode))
    monkeypatch.setattr("builtins.input", scripted_input(["pergunta", "sair"]))

    chat.main()

    output = capsys.readouterr().out
    assert "Erro ao consultar: could not connect to server" in output
    assert "Traceback" not in output
    assert "Até mais!" in output
