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
import time
from datetime import date
from pathlib import Path


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError(f"Não foi possível extrair o video_id da URL: {url}")


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def progress_bar(current: int, total: int, width: int = 28) -> str:
    filled = int(width * current / total)
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * current / total)
    return f"[{bar}] {pct:3d}%"


def banner(step: int, total_steps: int, label: str, overall_elapsed: float) -> None:
    overall_bar = progress_bar(step - 1, total_steps)
    print(f"\n{'═' * 60}")
    print(f"  Passo {step}/{total_steps} — {label}")
    print(f"  Geral: {overall_bar}  ⏱ {fmt_duration(overall_elapsed)} decorrido")
    print(f"{'═' * 60}")


def run_step(step: int, total_steps: int, label: str, cmd: list[str],
            overall_start: float) -> tuple[int, float]:
    banner(step, total_steps, label, time.time() - overall_start)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    elapsed = time.time() - t0
    status = "✓" if result.returncode == 0 else "✗"
    print(f"\n  {status} Passo {step} concluído em {fmt_duration(elapsed)}")
    return result.returncode, elapsed


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

    # Determina quantos passos serão executados
    total_steps = 1
    if not args.skip_summary:
        total_steps += 1
    if not args.skip_summary and not args.skip_podcast:
        total_steps += 1

    timings: dict[str, float] = {}
    overall_start = time.time()

    print(f"\n{'═' * 60}")
    print(f"  PIPELINE YouTubeProcessor  —  {total_steps} passo(s)")
    print(f"  Vídeo: {video_id}")
    print(f"  Modelo: {args.model}  |  Idioma: {args.lang}")
    print(f"{'═' * 60}")

    # ── Passo 1: Transcrição ──────────────────────────────────────────
    rc, t = run_step(1, total_steps, "Extraindo transcrição", [
        python, str(script_dir / "extract_transcript.py"),
        args.url,
        "--lang", args.lang,
        "--output-dir", str(base_dir),
    ], overall_start)
    timings["Transcrição"] = t
    if rc != 0:
        print(f"\nERRO no passo 1 (exit code {rc}). Abortando.")
        sys.exit(rc)

    # Localiza o JSON gerado
    date_str = date.today().strftime("%Y.%m.%d")
    matches = sorted(base_dir.glob(f"{date_str} - */{video_id}_*.json"))
    if not matches:
        print(f"\nERRO: JSON não encontrado em {base_dir}/{date_str} - */")
        sys.exit(1)
    json_path = matches[0]
    print(f"  → {json_path}")

    if args.skip_summary:
        _print_summary(timings, overall_start, total_steps,
                       json_path=json_path)
        sys.exit(0)

    # ── Passo 2: Resumo ───────────────────────────────────────────────
    rc, t = run_step(2, total_steps, "Gerando resumo com Ollama", [
        python, str(script_dir / "summarize_transcript.py"),
        str(json_path),
        "--provider", "ollama",
        "--model", args.model,
    ], overall_start)
    timings["Resumo"] = t
    if rc != 0:
        print(f"\nERRO no passo 2 (exit code {rc}). Abortando.")
        sys.exit(rc)

    md_path = json_path.with_suffix(".md")
    if not md_path.exists():
        print(f"\nERRO: Markdown não encontrado em: {md_path}")
        sys.exit(1)
    print(f"  → {md_path}")

    if args.skip_podcast:
        _print_summary(timings, overall_start, total_steps,
                       json_path=json_path, md_path=md_path)
        sys.exit(0)

    # ── Passo 3: Podcast ──────────────────────────────────────────────
    rc, t = run_step(3, total_steps, "Gerando podcast com Edge TTS", [
        python, str(script_dir / "generate_podcast.py"),
        str(md_path),
        "--provider", "ollama",
        "--model", args.model,
        "--tts", "edge",
    ], overall_start)
    timings["Podcast"] = t
    if rc != 0:
        print(f"\nERRO no passo 3 (exit code {rc}).")
        sys.exit(rc)

    mp3_path = md_path.with_suffix(".mp3")
    _print_summary(timings, overall_start, total_steps,
                   json_path=json_path, md_path=md_path, mp3_path=mp3_path)


def _print_summary(
    timings: dict[str, float],
    overall_start: float,
    total_steps: int,
    json_path: Path | None = None,
    md_path: Path | None = None,
    mp3_path: Path | None = None,
) -> None:
    total = time.time() - overall_start
    bar = progress_bar(total_steps, total_steps)
    print(f"\n{'═' * 60}")
    print(f"  PIPELINE CONCLUÍDO  {bar}  ⏱ {fmt_duration(total)} total")
    print(f"{'═' * 60}")
    print(f"  Tempos por etapa:")
    for name, t in timings.items():
        pct = int(t / total * 100) if total > 0 else 0
        print(f"    {name:<15} {fmt_duration(t):>8}  ({pct:3d}%)")
    print()
    if json_path:
        print(f"  Transcrição: {json_path}")
    if md_path:
        print(f"  Resumo:      {md_path}")
    if mp3_path:
        print(f"  Podcast:     {mp3_path}")
    print()


if __name__ == "__main__":
    main()
