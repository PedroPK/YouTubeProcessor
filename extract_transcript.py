"""
extract_transcript.py

Extrai transcrições de vídeos do YouTube com timestamps.

Abordagem 1 (padrão): usa as legendas nativas do YouTube via youtube-transcript-api
  - Rápido, sem download de áudio
  - Tem timestamps por segmento
  - NÃO identifica falantes (YouTube não fornece essa info)

Abordagem 2 (--full): baixa o áudio com yt-dlp e usa Whisper + pyannote para
  - Transcrição offline e mais precisa
  - Timestamps por palavra/segmento
  - Identificação de falantes (diarização)
  - Requer token HuggingFace gratuito e mais tempo de processamento

Uso:
  python extract_transcript.py <URL_do_YouTube>
  python extract_transcript.py <URL_do_YouTube> --lang pt
  python extract_transcript.py <URL_do_YouTube> --langs pt en  # tenta pt, depois en
  python extract_transcript.py <URL_do_YouTube> --list-langs   # lista idiomas disponíveis
"""

import argparse
import json
import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def format_timestamp(seconds: float) -> str:
    """Converte segundos em formato HH:MM:SS,mmm (padrão SRT)."""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def extract_video_id(url: str) -> str:
    """Extrai o video_id de diferentes formatos de URL do YouTube."""
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/|live/)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Não foi possível extrair o video_id da URL: {url}")


# ---------------------------------------------------------------------------
# Abordagem 1: youtube-transcript-api
# ---------------------------------------------------------------------------

def fetch_transcript_api(video_id: str, langs: list[str]) -> tuple[str, list[dict]]:
    """
    Retorna (idioma_usado, lista_de_segmentos).
    Cada segmento: {'text': str, 'start': float, 'duration': float}
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
    except ImportError:
        print("Instalando youtube-transcript-api...")
        os.system(f"{sys.executable} -m pip install youtube-transcript-api -q")
        from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

    # v1.x: API baseada em instância
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    # Tenta os idiomas solicitados, depois qualquer um disponível
    try:
        transcript = transcript_list.find_transcript(langs)
    except NoTranscriptFound:
        # Fallback: pega a primeira legenda disponível
        try:
            transcript = next(iter(transcript_list))
            print(f"  Idiomas solicitados não encontrados. Usando: {transcript.language} ({transcript.language_code})")
        except StopIteration:
            raise RuntimeError("Nenhuma legenda disponível para este vídeo.")

    fetched = transcript.fetch()
    # Normaliza para dicts simples (v1.x retorna FetchedTranscript iterável de snippets)
    normalized = []
    for seg in fetched:
        if hasattr(seg, 'text'):
            normalized.append({'text': seg.text, 'start': seg.start, 'duration': seg.duration})
        else:
            normalized.append(dict(seg))
    return transcript.language_code, normalized


def list_available_langs(video_id: str) -> None:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        os.system(f"{sys.executable} -m pip install youtube-transcript-api -q")
        from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    print("\nIdiomas disponíveis:")
    for t in transcript_list:
        kind = "auto-gerada" if t.is_generated else "manual"
        print(f"  [{t.language_code}] {t.language} ({kind})")


# ---------------------------------------------------------------------------
# Saída em múltiplos formatos
# ---------------------------------------------------------------------------

def save_json(segments: list[dict], path: Path, meta: dict) -> None:
    data = {"meta": meta, "segments": segments}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  JSON salvo: {path}")


def save_txt(segments: list[dict], path: Path) -> None:
    """Formato legível: [HH:MM:SS] texto"""
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            ts = format_timestamp(seg["start"])
            speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
            f.write(f"[{ts}] {speaker}{seg['text']}\n")
    print(f"  TXT salvo:  {path}")


def save_srt(segments: list[dict], path: Path) -> None:
    """Formato SRT padrão."""
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["start"] + seg.get("duration", 2.0))
            speaker = f"[{seg['speaker']}] " if seg.get("speaker") else ""
            f.write(f"{i}\n{start} --> {end}\n{speaker}{seg['text']}\n\n")
    print(f"  SRT salvo:  {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_video_title(video_id: str) -> str:
    """Tenta obter o título do vídeo via yt-dlp (sem download)."""
    try:
        import yt_dlp
        ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return info.get("title", video_id)
    except Exception:
        return video_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrai transcrições de vídeos do YouTube.")
    parser.add_argument("url", help="URL do vídeo no YouTube")
    parser.add_argument("--lang", default=None, help="Idioma preferido (ex: pt, en, es)")
    parser.add_argument("--langs", nargs="+", default=None, help="Lista de idiomas em ordem de preferência")
    parser.add_argument("--list-langs", action="store_true", help="Lista idiomas disponíveis e encerra")
    parser.add_argument("--output-dir", default="./transcripts", help="Diretório de saída (padrão: ./transcripts)")
    parser.add_argument("--formats", nargs="+", default=["json", "txt", "srt"],
                        choices=["json", "txt", "srt"], help="Formatos de saída")
    args = parser.parse_args()

    video_id = extract_video_id(args.url)
    print(f"\nVideo ID: {video_id}")

    if args.list_langs:
        list_available_langs(video_id)
        return

    # Determina idiomas a tentar
    if args.langs:
        langs = args.langs
    elif args.lang:
        langs = [args.lang]
    else:
        langs = ["pt", "pt-BR", "en"]  # padrão

    print(f"Buscando transcrição (idiomas preferidos: {langs})...")
    lang_used, segments = fetch_transcript_api(video_id, langs)
    print(f"  {len(segments)} segmentos obtidos | idioma: {lang_used}")

    # Metadados
    print("Obtendo título do vídeo...")
    title = get_video_title(video_id)
    print(f"  Título: {title}")

    meta = {
        "video_id": video_id,
        "url": args.url,
        "title": title,
        "language": lang_used,
        "segment_count": len(segments),
        "duration_seconds": (segments[-1]["start"] + segments[-1].get("duration", 0)) if segments else 0,
    }

    # Salva arquivos
    date_str = date.today().strftime("%Y.%m.%d")
    subdir_name = sanitize_filename(f"{date_str} - {title[:60]}")
    output_dir = Path(args.output_dir) / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = sanitize_filename(f"{video_id}_{title[:50]}")
    base_path = output_dir / base_name

    print(f"\nSalvando em {output_dir}/")
    if "json" in args.formats:
        save_json(segments, base_path.with_suffix(".json"), meta)
    if "txt" in args.formats:
        save_txt(segments, base_path.with_suffix(".txt"))
    if "srt" in args.formats:
        save_srt(segments, base_path.with_suffix(".srt"))

    print(f"\nConcluído! Duração total: {str(timedelta(seconds=int(meta['duration_seconds'])))}")


if __name__ == "__main__":
    main()
