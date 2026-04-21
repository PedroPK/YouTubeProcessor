"""
process_video.py

Pipeline completo: YouTube → Transcrição → Resumo → Podcast (100% local)

Encadeia automaticamente:
  1. extract_transcript.py  → transcrição via legendas nativas do YouTube
  2. summarize_transcript.py → resumo estruturado via Ollama
  3. generate_podcast.py    → roteiro + áudio via Edge TTS (gratuito, sem API key)

Pré-requisitos locais:
  - Ollama rodando: ollama serve
  - Modelo disponível: ollama pull llama3.2
  - Edge TTS: pip install edge-tts
  - ffmpeg: brew install ffmpeg

Uso:
  python process_video.py <URL>
  python process_video.py <URL> --model llama3.1
  python process_video.py <URL> --lang en
  python process_video.py <URL> --skip-podcast
  python process_video.py <URL> --skip-summary
"""

import argparse
import re
import sys
import subprocess
from datetime import date
from pathlib import Path


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError(f"Não foi possível extrair o video_id da URL: {url}")


def banner(text: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {text}")
    print(f"{'═' * 60}")


def run(label: str, cmd: list[str]) -> int:
    banner(label)
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline completo: YouTube → Transcrição → Resumo → Podcast (local)"
    )
    parser.add_argument("url", help="URL do vídeo no YouTube")
    parser.add_argument("--lang", default="pt",
                        help="Idioma da transcrição (padrão: pt)")
    parser.add_argument("--model", default="llama3.2",
                        help="Modelo Ollama para resumo e roteiro (padrão: llama3.2)")
    parser.add_argument("--output-dir", default="./transcripts",
                        help="Diretório base de saída (padrão: ./transcripts)")
    parser.add_argument("--skip-summary", action="store_true",
                        help="Pula o resumo e o podcast (só extrai transcrição)")
    parser.add_argument("--skip-podcast", action="store_true",
                        help="Pula a geração do podcast (extrai transcrição e resumo)")
    args = parser.parse_args()

    python = sys.executable
    script_dir = Path(__file__).parent
    base_dir = Path(args.output_dir)

    try:
        video_id = extract_video_id(args.url)
    except ValueError as e:
        print(f"ERRO: {e}")
        sys.exit(1)

    # ── Passo 1: Transcrição ──────────────────────────────────────────
    rc = run("PASSO 1/3 — Extraindo transcrição", [
        python, str(script_dir / "extract_transcript.py"),
        args.url,
        "--lang", args.lang,
        "--output-dir", str(base_dir),
    ])
    if rc != 0:
        print(f"\nERRO no passo 1 (exit code {rc}). Abortando.")
        sys.exit(rc)

    # Localiza o JSON gerado (mesmo padrão de nomenclatura do extract_transcript.py)
    date_str = date.today().strftime("%Y.%m.%d")
    matches = sorted(base_dir.glob(f"{date_str} - */{video_id}_*.json"))
    if not matches:
        print(f"\nERRO: JSON não encontrado em {base_dir}/{date_str} - */")
        sys.exit(1)
    json_path = matches[0]
    print(f"\n  → JSON: {json_path}")

    if args.skip_summary:
        banner("PIPELINE CONCLUÍDO (só transcrição)")
        print(f"  Transcrição: {json_path}")
        sys.exit(0)

    # ── Passo 2: Resumo ───────────────────────────────────────────────
    rc = run("PASSO 2/3 — Gerando resumo com Ollama", [
        python, str(script_dir / "summarize_transcript.py"),
        str(json_path),
        "--provider", "ollama",
        "--model", args.model,
    ])
    if rc != 0:
        print(f"\nERRO no passo 2 (exit code {rc}). Abortando.")
        sys.exit(rc)

    md_path = json_path.with_suffix(".md")
    if not md_path.exists():
        print(f"\nERRO: Markdown não encontrado em: {md_path}")
        sys.exit(1)
    print(f"\n  → Resumo: {md_path}")

    if args.skip_podcast:
        banner("PIPELINE CONCLUÍDO (sem podcast)")
        print(f"  Transcrição: {json_path}")
        print(f"  Resumo:      {md_path}")
        sys.exit(0)

    # ── Passo 3: Podcast ──────────────────────────────────────────────
    rc = run("PASSO 3/3 — Gerando podcast com Edge TTS", [
        python, str(script_dir / "generate_podcast.py"),
        str(md_path),
        "--provider", "ollama",
        "--model", args.model,
        "--tts", "edge",
    ])
    if rc != 0:
        print(f"\nERRO no passo 3 (exit code {rc}).")
        sys.exit(rc)

    mp3_path = md_path.with_suffix(".mp3")

    banner("PIPELINE CONCLUÍDO")
    print(f"  Transcrição: {json_path}")
    print(f"  Resumo:      {md_path}")
    print(f"  Podcast:     {mp3_path}")
    print()


if __name__ == "__main__":
    main()
