"""
summarize_transcript.py

Gera um resumo estruturado em Markdown a partir de um JSON de transcrição.

Estratégia:
  1. Divide a transcrição em janelas de tempo (ex: 15 min cada)
  2. Envia cada janela ao LLM para extração de tópicos, discussões e decisões
  3. Sintetiza todos os resumos parciais em um documento Markdown final

Provedores de LLM suportados:
  --provider openai   Usa OpenAI API (gpt-4o-mini por padrão). Requer OPENAI_API_KEY.
  --provider ollama   Usa Ollama local (llama3 por padrão). Gratuito, sem internet.
  --provider anthropic Usa Anthropic API (claude-3-5-haiku). Requer ANTHROPIC_API_KEY.

Uso:
  python summarize_transcript.py transcripts/video.json
  python summarize_transcript.py transcripts/video.json --provider ollama --model llama3.2
  python summarize_transcript.py transcripts/video.json --provider openai --model gpt-4o
  python summarize_transcript.py transcripts/video.json --chunk-minutes 20 --output resumo.md
"""

import argparse
import json
import os
import sys
import textwrap
import time
from datetime import timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_ts(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_chunks(segments: list[dict], chunk_minutes: int) -> list[dict]:
    """
    Agrupa segmentos em janelas de `chunk_minutes` minutos.
    Retorna lista de {'start': float, 'end': float, 'text': str}
    """
    chunk_seconds = chunk_minutes * 60
    chunks = []
    current_texts = []
    current_start = segments[0]["start"] if segments else 0.0
    chunk_boundary = current_start + chunk_seconds

    for seg in segments:
        if seg["start"] >= chunk_boundary and current_texts:
            end = seg["start"]
            chunks.append({
                "start": current_start,
                "end": end,
                "text": " ".join(current_texts),
            })
            current_texts = []
            current_start = seg["start"]
            chunk_boundary = current_start + chunk_seconds
        current_texts.append(seg["text"])

    if current_texts:
        last = segments[-1]
        chunks.append({
            "start": current_start,
            "end": last["start"] + last.get("duration", 0),
            "text": " ".join(current_texts),
        })
    return chunks


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CHUNK_SYSTEM = """Você é um especialista em documentação de reuniões, audiências públicas e debates.
Sua missão é produzir uma análise DETALHADA e FIEL do trecho de transcrição fornecido.
REGRAS OBRIGATÓRIAS:
- Use sempre o nome real dos interlocutores (corrija erros óbvios de transcrição, ex: "inúrupta" → "ininterrupta").
- Atribua cada argumento ao seu interlocutor: "Fulano defendeu que...", "Beltrana respondeu que...".
- Preserve réplicas e tréplicas — não colapse uma troca em uma única frase.
- Seja específico: cite dados, valores, datas e nomes de projetos/leis mencionados.
- Responda SEMPRE em português do Brasil."""

CHUNK_USER_TEMPLATE = """Analise o trecho abaixo (de {start} a {end}) da transcrição "{title}".

Produzir uma análise detalhada com as seguintes seções:

### Participantes neste trecho
Liste cada pessoa que falou: **Nome** — cargo ou papel — posição/argumento central.

### Debate e Posições
Para cada ponto discutido, descreva quem defendeu o quê, em ordem cronológica.
Incorpore réplicas e contra-argumentos com atribuição explícita de nome.
Seja específico: mencione dados concretos, valores, projetos, leis citadas.

### Decisões e Encaminhamentos
Liste explicitamente quaisquer decisões, votações ou próximos passos mencionados.
Se não houver nenhum neste trecho, escreva: "Nenhum neste trecho."

Trecho ({start}–{end}):
{text}

Responda SOMENTE com a análise nas três seções acima, sem introdução nem conclusão."""

SYNTHESIS_SYSTEM = """Você é um especialista em documentação de reuniões e audiências públicas.
Produza documentos ricos, detalhados e bem estruturados em português do Brasil.
REGRAS OBRIGATÓRIAS:
- Preserve TODOS os participantes identificados e suas posições específicas.
- Nas seções de debate, mostre os argumentos de cada lado com atribuição de fala.
- Inclua réplicas e tréplicas — não collapse o debate em uma frase genérica.
- Cite diretamente quando houver frases marcantes.
- Seja tão detalhado quanto o material permite."""

SYNTHESIS_USER_TEMPLATE = """Com base nos resumos parciais abaixo, gere um documento Markdown DETALHADO e COMPLETO
para a transcrição de: "{title}" (duração total: {duration}).

Resumos parciais por trecho (em ordem cronológica):
{partial_summaries}

O documento final deve conter as seguintes seções, nesta ordem:

# {title}

## Metadados
{metadata_table}

## Resumo Executivo
(3-5 parágrafos sintetizando o conteúdo; mencione os principais participantes e suas posições)

## Participantes
(para cada participante identificado: nome, cargo/organização e posição/argumento principal — use tabela ou lista estruturada)

## Temas e Tópicos Abordados
(lista hierárquica dos temas; agrupe subtópicos relacionados)

## Debates e Posições
(para cada ponto de debate: descreva a posição de cada interlocutor, incluindo réplicas e tréplicas;
use subseções por tema de debate; atribua cada argumento ao seu interlocutor pelo nome)

## Decisões e Encaminhamentos
(lista numerada de decisões, conclusões ou próximos passos; se nenhum, indicar)

## Linha do Tempo
(tabela: Horário | Interlocutor | Tópico — um registro por evento relevante)

---
IMPORTANTE: Não simplifique em excesso. Preserve a riqueza dos debates e as atribuições de fala.
Responda APENAS com o Markdown, sem texto adicional antes ou depois."""


# ---------------------------------------------------------------------------
# LLM Providers
# ---------------------------------------------------------------------------

def call_openai(prompt_system: str, prompt_user: str, model: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        print("Instalando openai...")
        os.system(f"{sys.executable} -m pip install openai -q")
        from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY não configurada.\n"
            "  export OPENAI_API_KEY='sk-...'"
        )
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def call_ollama(prompt_system: str, prompt_user: str, model: str) -> str:
    try:
        import requests
    except ImportError:
        os.system(f"{sys.executable} -m pip install requests -q")
        import requests

    url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": prompt_user},
        ],
        "options": {"temperature": 0.3},
    }
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=(30, None))
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama não está rodando. Inicie com: ollama serve\n"
            "  E baixe o modelo com: ollama pull llama3.2"
        )
    # Acumula tokens do stream — evita timeout de leitura em respostas longas
    result = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        token = chunk.get("message", {}).get("content", "")
        result.append(token)
        if chunk.get("done"):
            break
    return "".join(result).strip()


def call_anthropic(prompt_system: str, prompt_user: str, model: str) -> str:
    try:
        import anthropic
    except ImportError:
        print("Instalando anthropic...")
        os.system(f"{sys.executable} -m pip install anthropic -q")
        import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY não configurada.\n"
            "  export ANTHROPIC_API_KEY='sk-ant-...'"
        )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=prompt_system,
        messages=[{"role": "user", "content": prompt_user}],
    )
    return message.content[0].text.strip()


PROVIDERS = {
    "openai": (call_openai, "gpt-4o-mini"),
    "ollama": (call_ollama, "llama3.1"),
    "anthropic": (call_anthropic, "claude-3-5-haiku-20241022"),
}


def llm_call(provider: str, model: str, prompt_system: str, prompt_user: str) -> str:
    fn, _ = PROVIDERS[provider]
    return fn(prompt_system, prompt_user, model)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_chunk_narrative(raw: str) -> dict:
    """Envolve a narrativa livre do chunk em um dict uniforme."""
    return {"narrative": raw.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera resumo estruturado em Markdown de uma transcrição.")
    parser.add_argument("input", help="Arquivo JSON de transcrição (gerado por extract_transcript.py)")
    parser.add_argument("--provider", default="openai", choices=list(PROVIDERS),
                        help="Provedor de LLM (padrão: openai)")
    parser.add_argument("--model", default=None,
                        help="Modelo a usar (padrão: varia por provider)")
    parser.add_argument("--chunk-minutes", type=int, default=None,
                        help="Tamanho de cada janela de análise em minutos (padrão: 15, ou 5 para ollama)")
    parser.add_argument("--output", default=None,
                        help="Arquivo de saída .md (padrão: mesmo nome do input)")
    args = parser.parse_args()

    # Resolve modelo padrão
    _, default_model = PROVIDERS[args.provider]
    model = args.model or default_model

    # Chunk size padrão: ollama usa janelas maiores (10 min) para ter contexto
    # suficiente para identificar interlocutores e trocas de debate
    if args.chunk_minutes is None:
        chunk_minutes = 10 if args.provider == "ollama" else 15
    else:
        chunk_minutes = args.chunk_minutes

    # Lê transcrição
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERRO: Arquivo não encontrado: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = data["meta"]
    segments = data["segments"]
    title = meta["title"]
    duration_str = str(timedelta(seconds=int(meta["duration_seconds"])))

    print(f"\nTranscrição: {title}")
    print(f"Duração: {duration_str} | Segmentos: {len(segments)}")
    print(f"Provedor: {args.provider} | Modelo: {model}")

    # Divide em chunks
    chunks = build_chunks(segments, chunk_minutes)
    print(f"\nDividida em {len(chunks)} janelas de ~{chunk_minutes} min cada\n")

    # Passo 1: resume cada chunk
    partial_summaries = []
    chunk_times: list[float] = []
    chunks_start = time.time()
    for i, chunk in enumerate(chunks, 1):
        start_fmt = format_ts(chunk["start"])
        end_fmt = format_ts(chunk["end"])
        filled = int((i - 1) / len(chunks) * 28)
        bar = "█" * filled + "░" * (28 - filled)
        pct = int((i - 1) / len(chunks) * 100)
        elapsed = time.time() - chunks_start
        avg = elapsed / (i - 1) if i > 1 else 0
        eta = avg * (len(chunks) - i + 1)
        eta_str = f"ETA {int(eta // 60):02d}:{int(eta % 60):02d}" if avg > 0 else "ETA --:--"
        print(f"  [{i:02d}/{len(chunks):02d}] [{bar}] {pct:3d}%  {start_fmt}→{end_fmt}  {eta_str}", flush=True)

        t0 = time.time()
        prompt = CHUNK_USER_TEMPLATE.format(
            start=start_fmt,
            end=end_fmt,
            title=title,
            text=chunk["text"],
        )
        raw = llm_call(args.provider, model, CHUNK_SYSTEM, prompt)
        parsed = parse_chunk_narrative(raw)
        parsed["_start"] = start_fmt
        parsed["_end"] = end_fmt
        partial_summaries.append(parsed)
        elapsed_chunk = time.time() - t0
        chunk_times.append(elapsed_chunk)
        print(f"        ✓ concluído em {elapsed_chunk:.0f}s")

    total_chunks_time = time.time() - chunks_start
    avg_chunk = total_chunks_time / len(chunks) if chunks else 0
    print(f"\n  [{'█' * 28}] 100%  {len(chunks)} janelas em {int(total_chunks_time // 60):02d}:{int(total_chunks_time % 60):02d}  (média {avg_chunk:.0f}s/janela)")

    # Serializa resumos parciais para o prompt de síntese
    summaries_text = ""
    for ps in partial_summaries:
        summaries_text += f"\n---\n## Trecho {ps['_start']} – {ps['_end']}\n"
        summaries_text += ps.get("narrative", "") + "\n"
    # Passo 2: síntese final
    print(f"\nSintetizando documento final ...", flush=True)
    t0_synth = time.time()
    metadata_table = (
        f"| Campo | Valor |\n|---|---|\n"
        f"| Fonte | [{meta['url']}]({meta['url']}) |\n"
        f"| Idioma | {meta.get('language', '?')} |\n"
        f"| Duração | {duration_str} |\n"
        f"| Segmentos | {meta['segment_count']} |\n"
        f"| Video ID | {meta['video_id']} |"
    )
    synthesis_prompt = SYNTHESIS_USER_TEMPLATE.format(
        title=title,
        duration=duration_str,
        partial_summaries=summaries_text,
        metadata_table=metadata_table,
    )

    markdown = llm_call(args.provider, model, SYNTHESIS_SYSTEM, synthesis_prompt)
    synth_time = time.time() - t0_synth
    print(f"  ✓ síntese concluída em {synth_time:.0f}s")

    # Salva output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix(".md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\nResumo salvo em: {output_path}")

    # Salva também os resumos parciais em JSON para reuso
    partials_path = input_path.with_name(input_path.stem + "_chunks.json")
    with open(partials_path, "w", encoding="utf-8") as f:
        json.dump(partial_summaries, f, ensure_ascii=False, indent=2)
    print(f"Resumos parciais:  {partials_path}")


if __name__ == "__main__":
    main()
