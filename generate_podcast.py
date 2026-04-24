"""
generate_podcast.py

Converte um resumo em Markdown (gerado por summarize_transcript.py ou manualmente)
em um áudio estilo podcast com dois apresentadores.

Pipeline:
  1. Lê o arquivo Markdown com o resumo
  2. Usa LLM para converter em roteiro de podcast (JSON com falas por apresentador)
  3. Para cada fala, gera áudio via TTS (Microsoft Edge TTS — gratuito e sem API key)
  4. Concatena os segmentos com ffmpeg, adiciona pausa entre falas
  5. Exporta o episódio final em MP3

Provedores TTS suportados:
  --tts edge      Microsoft Edge TTS — gratuito, excelente qualidade, sem API key (padrão)
  --tts openai    OpenAI TTS — pago, muito natural (requer OPENAI_API_KEY)

Vozes padrão (Edge TTS, pt-BR):
  Apresentador 1 (host): pt-BR-AntonioNeural   (masculino)
  Apresentador 2 (co-host): pt-BR-FranciscaNeural (feminino)

Outros provedores LLM suportados para gerar o roteiro: openai, anthropic, ollama

Pré-requisitos:
  pip install edge-tts imageio-ffmpeg audioop-lts
  brew install ffmpeg   (macOS)  |  apt install ffmpeg  (Ubuntu)

Uso:
  python generate_podcast.py transcripts/video.md
  python generate_podcast.py transcripts/video.md --provider anthropic
  python generate_podcast.py transcripts/video.md --tts openai
  python generate_podcast.py transcripts/video.md --output episodio.mp3
  python generate_podcast.py transcripts/video.md --voice1 pt-BR-FranciscaNeural --voice2 pt-BR-ThalitaNeural
  python generate_podcast.py transcripts/video.md --list-voices   # lista vozes Edge disponíveis
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import datetime
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MIN_WORDS_PER_FALA = 80  # falas abaixo disto serão expandidas automaticamente

SCRIPT_SYSTEM = """Você é um roteirista sênior especializado em podcasts brasileiros de análise política e econômica.
Crie roteiros RICOS, DETALHADOS e envolventes em português do Brasil.
CADA FALA DEVE SER UM DISCURSO COMPLETO E DESENVOLVIDO, nunca uma frase solta."""

SCRIPT_USER_TEMPLATE = """Transforme o resumo abaixo em um roteiro de podcast com dois apresentadores:

- **{host1}** (host principal): conduz a conversa, apresenta os temas, contextualiza, faz perguntas aprofundadas
- **{host2}** (co-host / analista): desenvolve os pontos com profundidade, traz dados, perspectiva crítica e nuances

REGRAS DO ROTÉIROS — GERAIS:
- Linguagem conversacional e natural — como se estivessem falando ao vivo
- Duração estimada de {duration_min} a {duration_max} minutos de áudio
- Começar com uma introdução que prenda a atenção do ouvinte
- Cobrir os principais temas, discussões e encaminhamentos do conteúdo
- Terminar com uma conclusão e chamada para reflexão
- NÃO mencionar que é um resumo gerado — tratar como análise editorial
- NÃO usar termos como "conforme o documento" ou "segundo o resumo"

REGRAS DE DURAÇÃO — OBRIGATÓRIAS:
- Cada item de "falas" representa uma vez que o apresentador fala consecutivamente.
- O campo "texto" de CADA FALA deve ter NO MÍNIMO 100 palavras (idealmente 150 a 250).
- NUNCA escreva uma fala de 1 ou 2 frases — isso tornaria o podcast superficial e sem valor.
- Quando {host1} apresenta um tema, deve contextualizá-lo em pelo menos 3 frases antes de passar a palavra.
- Quando {host2} responde, deve desenvolver o argumento COMPLETAMENTE — dados, exemplos, implicações — antes de concluir.
- Quando alguém faz uma pergunta, deve elaborar o contexto em 2–3 frases antes da pergunta em si.
- Cada "fala" deve ser AUTOCONTIDA: quem ouvir apenas aquele trecho deve entender o que está sendo dito.

Responda APENAS com um JSON válido no formato:
{{
  "titulo": "título do episódio",
  "descricao": "uma frase de descrição para as plataformas de podcast",
  "duracao_estimada": "X-Y min",
  "falas": [
    {{"speaker": "{host1}", "texto": "..."}},
    {{"speaker": "{host2}", "texto": "..."}},
    ...
  ]
}}

Resumo a transformar:
{markdown_content}"""


# ---------------------------------------------------------------------------
# LLM — reutiliza lógica de summarize_transcript.py
# ---------------------------------------------------------------------------

def llm_call(provider: str, model: str, system: str, user: str) -> str:
    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError:
            os.system(f"{sys.executable} -m pip install openai -q")
            from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY não configurada.\n  export OPENAI_API_KEY='sk-...'")
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()

    elif provider == "anthropic":
        try:
            import anthropic
        except ImportError:
            os.system(f"{sys.executable} -m pip install anthropic -q")
            import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY não configurada.\n  export ANTHROPIC_API_KEY='sk-ant-...'")
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model or "claude-3-5-haiku-20241022",
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text.strip()

    elif provider == "ollama":
        try:
            import requests
        except ImportError:
            os.system(f"{sys.executable} -m pip install requests -q")
            import requests
        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/chat"
        payload = {
            "model": model or "llama3.1",
            "stream": True,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": 0.7},
        }
        try:
            resp = requests.post(url, json=payload, stream=True, timeout=(30, None))
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama erro: {e}\n  Certifique-se que está rodando: ollama serve")
        result = []
        chars = 0
        t0_stream = time.time()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            result.append(token)
            chars += len(token)
            elapsed = time.time() - t0_stream
            print(f"\r  {chars:5d} chars recebidos  ⏱ {elapsed:.0f}s ", end="", flush=True)
            if chunk.get("done"):
                break
        print()  # quebra de linha após o contador
        return "".join(result).strip()

    raise ValueError(f"Provider desconhecido: {provider}")


EXPAND_SYSTEM = """Você é um roteirista de podcasts. Expanda a fala fornecida mantendo o tom conversacional."""

EXPAND_USER_TEMPLATE = """A fala abaixo de {speaker} em um podcast sobre "{titulo}" está muito curta ({words} palavras).
Expanda-a para pelo menos 3 parágrafos completos (mínimo 150 palavras), desenvolvendo os argumentos
com detalhes, exemplos e contexto. Mantenha a voz e o tom do personagem.
NÃO adicione nenhum texto fora da fala — responda APENAS com o texto expandido, sem aspas nem JSON.

Fala original:
{texto}

Contexto do podcast (trecho anterior e posterior):
{contexto}"""


def expand_short_falas(
    falas: list[dict],
    titulo: str,
    provider: str,
    model: str,
) -> list[dict]:
    """Expande falas abaixo de MIN_WORDS_PER_FALA palavras."""
    short = [i for i, f in enumerate(falas) if len(f["texto"].split()) < MIN_WORDS_PER_FALA]
    if not short:
        return falas

    print(f"\n  {len(short)} fala(s) abaixo de {MIN_WORDS_PER_FALA} palavras — expandindo...")
    result = list(falas)
    for idx, i in enumerate(short, 1):
        fala = result[i]
        words = len(fala["texto"].split())
        prev_text = result[i - 1]["texto"][-200:] if i > 0 else ""
        next_text = result[i + 1]["texto"][:200] if i < len(result) - 1 else ""
        contexto = f"Fala anterior: ...{prev_text}\nFala posterior: {next_text}..."
        print(f"  [{idx}/{len(short)}] Expandindo fala {i+1} de {fala['speaker']} ({words} palavras)...", flush=True)
        t0 = time.time()
        prompt = EXPAND_USER_TEMPLATE.format(
            speaker=fala["speaker"],
            titulo=titulo,
            words=words,
            texto=fala["texto"],
            contexto=contexto,
        )
        expanded = llm_call(provider, model, EXPAND_SYSTEM, prompt)
        # Limpa aspas ou prefixos que o modelo possa ter adicionado
        expanded = expanded.strip().strip('"').strip("'").strip()
        result[i] = {**fala, "texto": expanded}
        print(f"        ✓ {len(expanded.split())} palavras em {time.time() - t0:.0f}s")
    return result


def parse_script_json(raw: str) -> dict:
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract the first complete JSON object using brace matching
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"Nenhum JSON encontrado na resposta do LLM:\n{raw[:500]}")
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(raw[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(raw[start:i + 1])
    raise ValueError(f"JSON incompleto na resposta do LLM:\n{raw[:500]}")


# ---------------------------------------------------------------------------
# TTS — Edge TTS (gratuito) e OpenAI TTS
# ---------------------------------------------------------------------------

async def tts_edge_segment(text: str, voice: str, output_path: Path) -> None:
    """Gera áudio para um segmento usando Microsoft Edge TTS."""
    try:
        import edge_tts
    except ImportError:
        print("  Instalando edge-tts...")
        os.system(f"{sys.executable} -m pip install edge-tts -q")
        import edge_tts

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


async def tts_edge_batch(segments: list[dict], voice_map: dict, tmp_dir: Path) -> list[Path]:
    """Gera todos os segmentos de áudio em paralelo (Edge TTS)."""
    tasks = []
    paths = []
    for i, seg in enumerate(segments):
        voice = voice_map.get(seg["speaker"], list(voice_map.values())[0])
        out = tmp_dir / f"seg_{i:04d}.mp3"
        paths.append(out)
        tasks.append(tts_edge_segment(seg["texto"], voice, out))

    # Processa em lotes para não sobrecarregar a API
    batch_size = 5
    t0_tts = time.time()
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        done = min(i + batch_size, len(tasks))
        filled = int(done / len(tasks) * 28)
        bar = "█" * filled + "░" * (28 - filled)
        pct = int(done / len(tasks) * 100)
        elapsed = time.time() - t0_tts
        print(f"  [{bar}] {pct:3d}%  {done}/{len(tasks)} segmentos  ⏱ {elapsed:.0f}s")

    return paths


def tts_openai_segment(text: str, voice: str, output_path: Path, client) -> None:
    """Gera áudio para um segmento usando OpenAI TTS."""
    resp = client.audio.speech.create(model="tts-1", voice=voice, input=text)
    resp.stream_to_file(str(output_path))


def tts_openai_batch(segments: list[dict], voice_map: dict, tmp_dir: Path) -> list[Path]:
    """Gera todos os segmentos com OpenAI TTS sequencialmente."""
    try:
        from openai import OpenAI
    except ImportError:
        os.system(f"{sys.executable} -m pip install openai -q")
        from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não configurada.")
    client = OpenAI(api_key=api_key)

    paths = []
    for i, seg in enumerate(segments):
        voice = voice_map.get(seg["speaker"], "alloy")
        out = tmp_dir / f"seg_{i:04d}.mp3"
        paths.append(out)
        tts_openai_segment(seg["texto"], voice, out, client)
        print(f"  Segmento {i+1}/{len(segments)}: [{seg['speaker']}]")

    return paths


async def list_edge_voices_pt() -> None:
    try:
        import edge_tts
    except ImportError:
        os.system(f"{sys.executable} -m pip install edge-tts -q")
        import edge_tts

    voices = await edge_tts.list_voices()
    pt_voices = [v for v in voices if v["Locale"].startswith("pt")]
    print("\nVozes Edge TTS disponíveis para português:\n")
    for v in pt_voices:
        gender = v.get("Gender", "?")
        print(f"  {v['ShortName']:40s}  {gender:8s}  {v['Locale']}")


# ---------------------------------------------------------------------------
# Concatenação de áudio
# ---------------------------------------------------------------------------

def _get_ffmpeg_binary() -> str:
    """Retorna o caminho do binário ffmpeg: do sistema (PATH) ou do bundle imageio-ffmpeg."""
    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg não encontrado.\n"
            "  Instale com: brew install ffmpeg (macOS) | apt install ffmpeg (Ubuntu)\n"
            "  Ou adicione ao venv: pip install imageio-ffmpeg"
        )


def concatenate_audio(audio_paths: list[Path], output_path: Path, pause_ms: int = 400) -> None:
    """Une todos os segmentos de áudio usando ffmpeg diretamente (não requer ffprobe)."""
    import subprocess
    import tempfile as _tempfile

    ffmpeg = _get_ffmpeg_binary()

    # Gera um segmento de silêncio em MP3
    with _tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        silence_path = f.name
    subprocess.run(
        [ffmpeg, "-y", "-f", "lavfi",
         "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
         "-t", str(pause_ms / 1000),
         "-q:a", "9", "-acodec", "libmp3lame", silence_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
    )

    # Cria arquivo de lista para o concat demuxer do ffmpeg
    with _tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list_path = f.name
        for i, path in enumerate(audio_paths):
            f.write(f"file '{path}'\n")
            if i < len(audio_paths) - 1:
                f.write(f"file '{silence_path}'\n")

    try:
        subprocess.run(
            [ffmpeg, "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list_path,
             "-acodec", "libmp3lame", "-q:a", "2",
             str(output_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
    finally:
        os.unlink(concat_list_path)
        os.unlink(silence_path)


# ---------------------------------------------------------------------------
# Helpers reutilizáveis
# ---------------------------------------------------------------------------

def _generate_script(
    input_path: Path,
    provider: str,
    model: str | None,
    host1: str,
    host2: str,
    duration_min: int,
    duration_max: int,
) -> tuple[dict, float]:
    """Gera o roteiro do podcast via LLM. Retorna (script_dict, elapsed_seconds)."""
    markdown_content = input_path.read_text(encoding="utf-8")
    max_chars = 4_000 if provider == "ollama" else 40_000
    if len(markdown_content) > max_chars:
        markdown_content = markdown_content[:max_chars] + "\n\n[...conteúdo truncado...]"
        if provider == "ollama":
            print(f"  (resumo truncado para {max_chars} chars)")

    print("Gerando roteiro via LLM...")
    t0 = time.time()
    prompt = SCRIPT_USER_TEMPLATE.format(
        host1=host1,
        host2=host2,
        duration_min=duration_min,
        duration_max=duration_max,
        markdown_content=markdown_content,
    )
    raw = llm_call(provider, model, SCRIPT_SYSTEM, prompt)
    elapsed = time.time() - t0
    script = parse_script_json(raw)
    print(f"  ✓ roteiro concluído em {elapsed:.0f}s")

    script["falas"] = expand_short_falas(
        script.get("falas", []),
        titulo=script.get("titulo", ""),
        provider=provider,
        model=model,
    )
    return script, elapsed


def _generate_audio(
    falas: list[dict],
    voice_map: dict,
    tts: str,
    output_path: Path,
    pause_ms: int,
) -> None:
    """Gera e concatena o áudio para a lista de falas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        print(f"Gerando áudio para {len(falas)} segmentos ({tts})...")

        if tts == "edge":
            audio_paths = asyncio.run(tts_edge_batch(falas, voice_map, tmp))
        else:
            audio_paths = tts_openai_batch(falas, voice_map, tmp)

        missing = [p for p in audio_paths if not p.exists()]
        if missing:
            print(f"AVISO: {len(missing)} segmentos não puderam ser gerados.")
            audio_paths = [p for p in audio_paths if p.exists()]

        print(f"\nConcatenando {len(audio_paths)} segmentos (pausa: {pause_ms}ms)...", end=" ", flush=True)
        concatenate_audio(audio_paths, output_path, pause_ms=pause_ms)
        print("ok")


def _script_stats(script: dict) -> dict:
    """Calcula estatísticas de palavras por fala."""
    falas = script.get("falas", [])
    if not falas:
        return {"total_falas": 0, "min_words": 0, "max_words": 0, "avg_words": 0,
                "total_words": 0, "short_falas": 0}
    word_counts = [len(f["texto"].split()) for f in falas]
    return {
        "total_falas": len(falas),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "avg_words": int(sum(word_counts) / len(word_counts)),
        "total_words": sum(word_counts),
        "short_falas": sum(1 for w in word_counts if w < MIN_WORDS_PER_FALA),
    }


def _write_comparison_md(
    results: list[dict],
    input_path: Path,
    provider: str,
    total_elapsed: float,
    script_only: bool,
) -> Path:
    """Gera arquivo Markdown com a comparação dos modelos."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    models = [r["model"] for r in results]
    out_path = input_path.with_name(f"{input_path.stem}_comparison.md")

    lines: list[str] = []
    lines.append(f"# Comparação de Modelos — {input_path.stem}")
    lines.append(f"")
    lines.append(f"**Data:** {now}  ")
    lines.append(f"**Provider:** {provider}  ")
    lines.append(f"**Modelos:** {', '.join(models)}  ")
    lines.append(f"**Tempo total:** {int(total_elapsed // 60):02d}:{int(total_elapsed % 60):02d}  ")
    lines.append(f"")
    lines.append("---")
    lines.append(f"")

    # ── Tabela de métricas ──────────────────────────────────────────
    lines.append("## Métricas comparativas")
    lines.append(f"")
    header = "| Métrica | " + " | ".join(models) + " |"
    sep    = "|---|" + "---|" * len(models)
    lines.append(header)
    lines.append(sep)

    rows = [
        ("Falas geradas",         lambda r: str(r["total_falas"])),
        ("Total de palavras",     lambda r: str(r["total_words"])),
        ("Palavras/fala (média)", lambda r: str(r["avg_words"])),
        ("Palavras/fala (mín)",  lambda r: str(r["min_words"])),
        ("Palavras/fala (máx)",  lambda r: str(r["max_words"])),
        ("Falas curtas (<80p)",  lambda r: str(r["short_falas"])),
        ("Duração estimada",      lambda r: r["duracao_estimada"]),
        ("Tempo roteiro",         lambda r: f"{r['script_time']:.0f}s"),
        ("Tempo total",           lambda r: f"{r['total_time']:.0f}s"),
        ("Tamanho MP3",           lambda r: f"{r['size_mb']:.1f} MB" if r["size_mb"] else "—"),
    ]
    for label, fn in rows:
        lines.append("| " + label + " | " + " | ".join(fn(r) for r in results) + " |")
    lines.append(f"")

    # ── Títulos gerados ─────────────────────────────────────────────
    lines.append("## Títulos gerados")
    lines.append(f"")
    for r in results:
        lines.append(f"**{r['model']}:** {r['titulo']}  ")
    lines.append(f"")

    # ── Roteiros por modelo ─────────────────────────────────────────
    for r in results:
        safe = r["model"].replace(":", "_").replace("/", "_")
        script_path = input_path.with_name(f"{input_path.stem}_script_{safe}.json")
        try:
            with open(script_path, encoding="utf-8") as f:
                script = json.load(f)
            falas = script.get("falas", [])
        except Exception:
            falas = []

        lines.append(f"---")
        lines.append(f"")
        lines.append(f"## Roteiro: {r['model']}")
        lines.append(f"")
        lines.append(f"**Título:** {r['titulo']}  ")
        lines.append(f"**Duração estimada:** {r['duracao_estimada']}  ")
        lines.append(f"**Falas:** {r['total_falas']}  |  **Palavras:** {r['total_words']}  |  **Média/fala:** {r['avg_words']}  ")
        if not script_only:
            mp3 = r.get("mp3_path")
            if mp3 and Path(mp3).exists():
                lines.append(f"**Áudio:** `{Path(mp3).name}`  ")
        lines.append(f"")

        for i, fala in enumerate(falas, 1):
            host = fala.get("speaker", fala.get("host", fala.get("apresentador", "?")))
            text = fala.get("texto", fala.get("text", fala.get("fala", "")))
            lines.append(f"**{host}:** {text}  ")
            lines.append(f"")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    """Retorna string com barra de progresso: [████░░░░] XX%"""
    filled = int(width * current / total) if total else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total) if total else 0
    return f"[{bar}] {pct:3d}%"


def _ollama_unload(model: str) -> None:
    """Descarrega o modelo da RAM do Ollama (keep_alive=0) para liberar memória."""
    try:
        import requests as _req
        url = os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
        _req.post(url, json={"model": model, "keep_alive": 0}, timeout=10)
        print(f"  ↓ modelo '{model}' descarregado da RAM")
    except Exception:
        pass  # não crítico, apenas uma otimização de memória


def _run_comparison(args, input_path: Path, voice_map: dict, models: list[str]) -> None:
    """Gera roteiro e áudio para cada modelo SEQUENCIALMENTE e exibe tabela comparativa."""
    results = []
    total = len(models)
    comparison_start = time.time()
    print(f"\n{'═' * 60}")
    print(f"  MODO COMPARAÇÃO: {total} modelos (sequencial)")
    print(f"  {' → '.join(models)}")
    print(f"  Cada modelo será descarregado da RAM antes do próximo iniciar.")
    print(f"{'═' * 60}")

    for idx, model in enumerate(models, 1):
        safe_name = model.replace(":", "_").replace("/", "_")
        script_path = input_path.with_name(f"{input_path.stem}_script_{safe_name}.json")
        mp3_path = input_path.with_name(f"{input_path.stem}_{safe_name}.mp3")

        # Barra de progresso geral
        elapsed_total = time.time() - comparison_start
        if idx > 1 and elapsed_total > 0:
            avg_per_model = elapsed_total / (idx - 1)
            eta_s = avg_per_model * (total - idx + 1)
            eta_str = f"  ETA {int(eta_s // 60):02d}:{int(eta_s % 60):02d}"
        else:
            eta_str = ""
        bar_str = _progress_bar(idx - 1, total)
        print(f"\n{'─' * 60}")
        print(f"  {bar_str}  modelo {idx}/{total}{eta_str}")
        print(f"  Processando: {model}")
        print(f"{'─' * 60}")

        t_total = time.time()
        script, script_elapsed = _generate_script(
            input_path=input_path,
            provider=args.provider,
            model=model,
            host1=args.host1,
            host2=args.host2,
            duration_min=args.duration_min,
            duration_max=args.duration_max,
        )

        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        print(f"  Roteiro: {script_path}")

        if not args.script_only:
            _generate_audio(script.get("falas", []), voice_map, args.tts, mp3_path, args.pause_ms)
            size_mb = mp3_path.stat().st_size / 1_000_000 if mp3_path.exists() else 0
            print(f"  Áudio:   {mp3_path}  ({size_mb:.1f} MB)")
        else:
            size_mb = 0

        stats = _script_stats(script)
        model_elapsed = time.time() - t_total
        results.append({
            "model": model,
            "titulo": script.get("titulo", "—"),
            "duracao_estimada": script.get("duracao_estimada", "—"),
            "script_time": script_elapsed,
            "total_time": model_elapsed,
            "size_mb": size_mb,
            "script_path": script_path,
            "mp3_path": mp3_path if not args.script_only else None,
            **stats,
        })

        done_bar = _progress_bar(idx, total)
        print(f"  {done_bar}  modelo {idx}/{total}  ✓ concluído em {model_elapsed:.0f}s")

        # Descarrega modelo da RAM antes de passar para o próximo
        if args.provider == "ollama" and idx < total:
            _ollama_unload(model)

    # ── Tabela comparativa ──────────────────────────────────────────
    total_elapsed = time.time() - comparison_start
    print(f"\n{'═' * 60}")
    print(f"  RESULTADO DA COMPARAÇÃO")
    print(f"  {_progress_bar(total, total)}  {total}/{total} modelos  ✓ total {int(total_elapsed // 60):02d}:{int(total_elapsed % 60):02d}")
    print(f"{'═' * 60}")
    col = 22
    header = f"  {'Métrica':<24}" + "".join(f"{m:>{col}}" for m in models)
    print(header)
    print(f"  {'─' * (24 + col * len(models))}")

    rows = [
        ("Falas geradas",        lambda r: str(r["total_falas"])),
        ("Total de palavras",    lambda r: str(r["total_words"])),
        ("Palavras/fala (média)", lambda r: str(r["avg_words"])),
        ("Palavras/fala (mín)",  lambda r: str(r["min_words"])),
        ("Palavras/fala (máx)",  lambda r: str(r["max_words"])),
        ("Falas curtas (<80p)",  lambda r: str(r["short_falas"])),
        ("Duração estimada",     lambda r: r["duracao_estimada"]),
        ("Tempo roteiro",        lambda r: f"{r['script_time']:.0f}s"),
        ("Tempo total",          lambda r: f"{r['total_time']:.0f}s"),
        ("Tamanho MP3",          lambda r: f"{r['size_mb']:.1f} MB" if r["size_mb"] else "—"),
    ]
    for label, fn in rows:
        print(f"  {label:<24}" + "".join(f"{fn(r):>{col}}" for r in results))

    print(f"\n  Títulos gerados:")
    for r in results:
        print(f"    [{r['model']}]  {r['titulo']}")

    print(f"\n  Roteiros JSON salvos:")
    for r in results:
        print(f"    {r['script_path']}")
    if not args.script_only:
        print(f"\n  Áudios MP3 salvos:")
        for r in results:
            print(f"    {r['mp3_path']}")

    md_path = _write_comparison_md(results, input_path, args.provider, total_elapsed, args.script_only)
    print(f"\n  Documentação comparativa: {md_path}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Gera podcast em áudio a partir de um resumo Markdown.")
    parser.add_argument("input", help="Arquivo Markdown (.md) com o resumo a converter")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "ollama"],
                        help="Provedor LLM para gerar o roteiro (padrão: openai)")
    parser.add_argument("--model", default=None, help="Modelo LLM (padrão: varia por provider)")
    parser.add_argument("--tts", default="edge", choices=["edge", "openai"],
                        help="Provedor TTS para síntese de voz (padrão: edge)")
    parser.add_argument("--voice1", default="pt-BR-AntonioNeural",
                        help="Voz do host principal (padrão: pt-BR-AntonioNeural)")
    parser.add_argument("--voice2", default="pt-BR-FranciscaNeural",
                        help="Voz do co-host (padrão: pt-BR-FranciscaNeural)")
    parser.add_argument("--host1", default="Rafael", help="Nome do host principal (padrão: Rafael)")
    parser.add_argument("--host2", default="Ana", help="Nome do co-host (padrão: Ana)")
    parser.add_argument("--duration-min", type=int, default=8,
                        help="Duração mínima estimada do podcast em minutos (padrão: 8)")
    parser.add_argument("--duration-max", type=int, default=12,
                        help="Duração máxima estimada do podcast em minutos (padrão: 12)")
    parser.add_argument("--pause-ms", type=int, default=400,
                        help="Pausa entre falas em milissegundos (padrão: 400)")
    parser.add_argument("--output", default=None, help="Arquivo MP3 de saída")
    parser.add_argument("--script-only", action="store_true",
                        help="Gera apenas o roteiro JSON sem produzir áudio")
    parser.add_argument("--from-script", default=None,
                        help="Pula geração do roteiro e usa um JSON existente diretamente")
    parser.add_argument("--list-voices", action="store_true",
                        help="Lista vozes Edge TTS disponíveis para português e encerra")
    parser.add_argument("--compare-models", default=None, metavar="MODELO_A,MODELO_B",
                        help="Gera roteiro e áudio com dois modelos e exibe tabela comparativa. "
                             "Ex: --compare-models llama3.1,llama3.2")
    args = parser.parse_args()

    # Lista vozes
    if args.list_voices:
        asyncio.run(list_edge_voices_pt())
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERRO: Arquivo não encontrado: {input_path}")
        sys.exit(1)

    # Mapa de vozes por speaker
    if args.tts == "edge":
        voice_map = {args.host1: args.voice1, args.host2: args.voice2}
    else:
        voice_map = {args.host1: "onyx", args.host2: "nova"}

    # Modo comparação
    if args.compare_models:
        models = [m.strip() for m in args.compare_models.split(",") if m.strip()]
        if len(models) < 2:
            print("ERRO: --compare-models requer dois modelos separados por vírgula.")
            sys.exit(1)
        _run_comparison(args, input_path, voice_map, models)
        return

    output_path = Path(args.output) if args.output else input_path.with_suffix(".mp3")
    script_path = input_path.with_name(input_path.stem + "_podcast_script.json")

    # --------------- Passo 1: Roteiro ---------------
    if args.from_script:
        print(f"\nUsando roteiro existente: {args.from_script}")
        with open(args.from_script, encoding="utf-8") as f:
            podcast_script = json.load(f)
    else:
        podcast_script, _ = _generate_script(
            input_path=input_path,
            provider=args.provider,
            model=args.model,
            host1=args.host1,
            host2=args.host2,
            duration_min=args.duration_min,
            duration_max=args.duration_max,
        )
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(podcast_script, f, ensure_ascii=False, indent=2)
        print(f"Roteiro salvo: {script_path}")
        print(f"\nTítulo:   {podcast_script.get('titulo', '—')}")
        print(f"Descrição: {podcast_script.get('descricao', '—')}")
        print(f"Duração:  {podcast_script.get('duracao_estimada', '—')}")
        print(f"Falas:    {len(podcast_script.get('falas', []))} segmentos\n")

    if args.script_only:
        print("Modo --script-only: áudio não gerado.")
        return

    falas = podcast_script.get("falas", [])
    if not falas:
        print("ERRO: Roteiro sem falas.")
        sys.exit(1)

    # --------------- Passo 2 + 3: TTS + Concatenação ---------------
    _generate_audio(falas, voice_map, args.tts, output_path, args.pause_ms)
    size_mb = output_path.stat().st_size / 1_000_000
    print(f"\nPodcast gerado: {output_path}  ({size_mb:.1f} MB)")
    print(f"Roteiro JSON:   {script_path}")


if __name__ == "__main__":
    main()
