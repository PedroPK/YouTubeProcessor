"""
diarize_transcript.py

Abordagem 2: Download de áudio + Whisper (transcrição) + pyannote (diarização de falantes)

Identifica QUEM fala e QUANDO — algo que as legendas nativas do YouTube não fornecem.

Pré-requisitos:
  1. Instalar dependências pesadas:
       pip install openai-whisper pyannote.audio yt-dlp torch torchaudio

  2. Aceitar as condições de uso dos modelos pyannote no HuggingFace:
       https://huggingface.co/pyannote/speaker-diarization-3.1
       https://huggingface.co/pyannote/segmentation-3.0
     (requer conta gratuita e aceitar os termos de cada modelo)

  3. Gerar um token de acesso no HuggingFace:
       https://hf.co/settings/tokens
     e setar como variável de ambiente:
       export HF_TOKEN="hf_xxxxxxxxxxxx"

Uso:
  python diarize_transcript.py <URL_do_YouTube>
  python diarize_transcript.py <URL_do_YouTube> --model medium --output-dir ./transcripts

Modelos Whisper disponíveis (menor = mais rápido, maior = mais preciso):
  tiny, base, small, medium, large, large-v2, large-v3
  Recomendação para português: medium ou large-v2

Tempo estimado (CPU, vídeo de 1h):
  - Download: ~2-5 min
  - Transcrição (medium): ~20-40 min
  - Diarização: ~5-15 min
  Com GPU (CUDA): 5-10x mais rápido
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import date, timedelta


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    millis = int((seconds - int(seconds)) * 1000)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"


def extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|youtu\.be/|embed/|shorts/|live/)([a-zA-Z0-9_-]{11})", url)
    if match:
        return match.group(1)
    raise ValueError(f"Não foi possível extrair o video_id da URL: {url}")


def download_audio(url: str, output_path: Path) -> Path:
    """Baixa apenas o áudio do vídeo em formato wav (melhor para Whisper)."""
    import yt_dlp
    audio_file = output_path.with_suffix(".wav")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path.with_suffix("")),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "video")
    return audio_file, title


def transcribe_whisper(audio_path: Path, model_name: str = "medium") -> dict:
    """Transcreve com OpenAI Whisper. Retorna resultado com segmentos e timestamps."""
    import whisper
    print(f"  Carregando modelo Whisper '{model_name}'...")
    model = whisper.load_model(model_name)
    print("  Transcrevendo (isso pode demorar)...")
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,  # timestamps por palavra
        verbose=False,
    )
    return result


def diarize_audio(audio_path: Path, hf_token: str) -> list[dict]:
    """
    Identifica os intervalos de fala de cada falante via pyannote.
    Retorna lista de {'speaker': 'SPEAKER_00', 'start': float, 'end': float}
    """
    from pyannote.audio import Pipeline
    print("  Carregando pipeline de diarização...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    print("  Diarizando áudio (identificando falantes)...")
    diarization = pipeline(str(audio_path))
    turns = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({"speaker": speaker, "start": segment.start, "end": segment.end})
    return turns


def assign_speakers_to_segments(whisper_segments: list, diarization_turns: list) -> list[dict]:
    """
    Combina os segmentos do Whisper com os intervalos de diarização.
    Cada segmento recebe o falante dominante no seu intervalo de tempo.
    """
    result = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        # Calcula sobreposição com cada turno de fala
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        for turn in diarization_turns:
            overlap_start = max(seg_start, turn["start"])
            overlap_end = min(seg_end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        result.append({
            "text": seg["text"].strip(),
            "start": seg_start,
            "duration": seg_end - seg_start,
            "speaker": best_speaker,
        })
    return result


def save_outputs(segments: list[dict], meta: dict, output_dir: Path, base_name: str, formats: list[str]) -> None:
    base_path = output_dir / base_name

    if "json" in formats:
        data = {"meta": meta, "segments": segments}
        path = base_path.with_suffix(".json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  JSON salvo: {path}")

    if "txt" in formats:
        path = base_path.with_suffix(".txt")
        with open(path, "w", encoding="utf-8") as f:
            current_speaker = None
            for seg in segments:
                ts = format_timestamp(seg["start"])
                speaker = seg.get("speaker", "")
                if speaker != current_speaker:
                    f.write(f"\n[{speaker}]\n")
                    current_speaker = speaker
                f.write(f"  [{ts}] {seg['text']}\n")
        print(f"  TXT salvo:  {path}")

    if "srt" in formats:
        path = base_path.with_suffix(".srt")
        with open(path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start = format_timestamp(seg["start"])
                end = format_timestamp(seg["start"] + seg.get("duration", 2.0))
                speaker = seg.get("speaker", "")
                f.write(f"{i}\n{start} --> {end}\n[{speaker}] {seg['text']}\n\n")
        print(f"  SRT salvo:  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcrição com identificação de falantes.")
    parser.add_argument("url", help="URL do vídeo no YouTube")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        help="Modelo Whisper a usar (padrão: medium)")
    parser.add_argument("--output-dir", default="./transcripts")
    parser.add_argument("--formats", nargs="+", default=["json", "txt", "srt"],
                        choices=["json", "txt", "srt"])
    parser.add_argument("--hf-token", default=None,
                        help="Token HuggingFace (ou set HF_TOKEN env var)")
    parser.add_argument("--no-diarization", action="store_true",
                        help="Pula a diarização (apenas transcrição Whisper)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not args.no_diarization and not hf_token:
        print("ERRO: Token HuggingFace necessário para diarização.")
        print("  Opção 1: --hf-token hf_xxxx")
        print("  Opção 2: export HF_TOKEN=hf_xxxx")
        print("  Opção 3: usar --no-diarization para pular a identificação de falantes")
        sys.exit(1)

    video_id = extract_video_id(args.url)
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = base_output_dir / f"{video_id}_audio"

    print(f"\nVideo ID: {video_id}")
    print("Baixando áudio...")
    audio_file, title = download_audio(args.url, audio_path)

    date_str = date.today().strftime("%Y.%m.%d")
    subdir_name = sanitize_filename(f"{date_str} - {title[:60]}")
    output_dir = base_output_dir / subdir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Áudio: {audio_file} ({audio_file.stat().st_size / 1_000_000:.1f} MB)")

    print(f"\nTranscrevendo com Whisper ({args.model})...")
    whisper_result = transcribe_whisper(audio_file, args.model)
    segments_raw = whisper_result["segments"]
    print(f"  {len(segments_raw)} segmentos transcritos")

    if not args.no_diarization:
        print("\nDiarizando falantes...")
        diarization_turns = diarize_audio(audio_file, hf_token)
        print(f"  {len(set(t['speaker'] for t in diarization_turns))} falantes detectados")
        segments = assign_speakers_to_segments(segments_raw, diarization_turns)
    else:
        segments = [
            {"text": s["text"].strip(), "start": s["start"],
             "duration": s["end"] - s["start"]}
            for s in segments_raw
        ]

    meta = {
        "video_id": video_id,
        "url": args.url,
        "title": title,
        "whisper_model": args.model,
        "language": whisper_result.get("language", "unknown"),
        "diarization": not args.no_diarization,
        "segment_count": len(segments),
        "duration_seconds": segments[-1]["start"] + segments[-1].get("duration", 0) if segments else 0,
    }

    base_name = sanitize_filename(f"{video_id}_{title[:50]}_diarized")
    print(f"\nSalvando em {output_dir}/")
    save_outputs(segments, meta, output_dir, base_name, args.formats)

    # Limpa arquivo de áudio temporário
    audio_file.unlink(missing_ok=True)
    print(f"\nConcluído! Duração: {str(timedelta(seconds=int(meta['duration_seconds'])))}")
    if not args.no_diarization:
        speakers = set(s.get("speaker", "") for s in segments if s.get("speaker"))
        print(f"Falantes identificados: {', '.join(sorted(speakers))}")
        print("(renomeie os falantes SPEAKER_00, SPEAKER_01... conforme necessário)")


if __name__ == "__main__":
    main()
