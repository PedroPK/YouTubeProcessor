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
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SCRIPT_SYSTEM = """Você é um roteirista especializado em podcasts brasileiros de análise política e econômica.
Crie roteiros naturais, conversacionais e envolventes em português do Brasil."""

SCRIPT_USER_TEMPLATE = """Transforme o resumo abaixo em um roteiro de podcast com dois apresentadores:

- **{host1}** (host principal): conduz a conversa, apresenta os temas, faz perguntas
- **{host2}** (co-host / analista): aprofunda os pontos, dá contexto, traz perspectiva crítica

Regras do roteiro:
- Linguagem conversacional e natural — como se estivessem falando ao vivo
- Duração estimada de {duration_min} a {duration_max} minutos de áudio
- Começar com uma introdução que prenda a atenção do ouvinte
- Cobrir os principais temas, discussões e encaminhamentos do conteúdo
- Terminar com uma conclusão e chamada para reflexão
- NÃO mencionar que é um resumo gerado — tratar como análise editorial
- NÃO usar termos como "conforme o documento" ou "segundo o resumo"

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
    args = parser.parse_args()

    # Lista vozes
    if args.list_voices:
        asyncio.run(list_edge_voices_pt())
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERRO: Arquivo não encontrado: {input_path}")
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".mp3")
    script_path = input_path.with_name(input_path.stem + "_podcast_script.json")

    # Mapa de vozes por speaker
    if args.tts == "edge":
        voice_map = {args.host1: args.voice1, args.host2: args.voice2}
    else:
        # Vozes OpenAI: alloy, echo, fable, onyx, nova, shimmer
        voice_map = {args.host1: "onyx", args.host2: "nova"}

    # --------------- Passo 1: Roteiro ---------------
    if args.from_script:
        print(f"\nUsando roteiro existente: {args.from_script}")
        with open(args.from_script, encoding="utf-8") as f:
            podcast_script = json.load(f)
    else:
        markdown_content = input_path.read_text(encoding="utf-8")

        # Limita o tamanho para caber no contexto do LLM
        # Modelos locais (ollama) são mais lentos: usa limite menor
        # Modelos 1B (ex: llama3.2:1b) precisam de contexto ainda menor
        if args.provider == "ollama":
            max_chars = 4_000
        else:
            max_chars = 40_000
        if len(markdown_content) > max_chars:
            markdown_content = markdown_content[:max_chars] + "\n\n[...conteúdo truncado...]"
            if args.provider == "ollama":
                print(f"  (resumo truncado para {max_chars} chars para melhorar velocidade no modelo local)")

        print(f"\nArquivo: {input_path.name}")
        print(f"Provider LLM: {args.provider} | TTS: {args.tts}")
        print(f"Hosts: {args.host1} ({args.voice1}) e {args.host2} ({args.voice2})")
        print(f"Duração alvo: {args.duration_min}–{args.duration_max} min\n")

        print("Gerando roteiro de podcast via LLM...")
        t0_script = time.time()
        prompt = SCRIPT_USER_TEMPLATE.format(
            host1=args.host1,
            host2=args.host2,
            duration_min=args.duration_min,
            duration_max=args.duration_max,
            markdown_content=markdown_content,
        )
        raw = llm_call(args.provider, args.model, SCRIPT_SYSTEM, prompt)
        script_time = time.time() - t0_script
        podcast_script = parse_script_json(raw)
        print(f"  ✓ roteiro concluído em {script_time:.0f}s")

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

    # --------------- Passo 2: TTS ---------------
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        print(f"Gerando áudio para {len(falas)} segmentos ({args.tts})...")

        if args.tts == "edge":
            audio_paths = asyncio.run(tts_edge_batch(falas, voice_map, tmp))
        else:
            audio_paths = tts_openai_batch(falas, voice_map, tmp)

        # Verifica se todos foram gerados
        missing = [p for p in audio_paths if not p.exists()]
        if missing:
            print(f"AVISO: {len(missing)} segmentos não puderam ser gerados.")
            audio_paths = [p for p in audio_paths if p.exists()]

        # --------------- Passo 3: Concatenação ---------------
        print(f"\nConcatenando {len(audio_paths)} segmentos (pausa: {args.pause_ms}ms)...", end=" ", flush=True)
        concatenate_audio(audio_paths, output_path, pause_ms=args.pause_ms)
        print("ok")

    size_mb = output_path.stat().st_size / 1_000_000
    print(f"\nPodcast gerado: {output_path}  ({size_mb:.1f} MB)")
    print(f"Roteiro JSON:   {script_path}")


if __name__ == "__main__":
    main()
