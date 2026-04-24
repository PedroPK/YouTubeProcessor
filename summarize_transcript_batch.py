"""
summarize_transcript_batch.py

Gera resumos executivos em Markdown para vários modelos a partir de uma única transcrição JSON.

Uso:
  python summarize_transcript_batch.py transcripts/video.json --provider ollama --models llama3.1,gemma3:4b,mistral

Saída:
  transcripts/video_llama3.1_summary.md
  transcripts/video_gemma3_4b_summary.md
  transcripts/video_mistral_summary.md
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera resumos executivos em Markdown para vários modelos a partir de uma transcrição JSON."
    )
    parser.add_argument("input", help="Arquivo JSON de transcrição gerado por extract_transcript.py")
    parser.add_argument("--provider", default="ollama", choices=["openai", "anthropic", "ollama"],
                        help="Provedor LLM para gerar os resumos (padrão: ollama)")
    parser.add_argument("--models", required=True,
                        help="Lista de modelos separados por vírgula, exemplo: llama3.1,gemma3:4b,mistral")
    parser.add_argument("--chunk-minutes", type=int, default=None,
                        help="Tamanho de janela para summarize_transcript.py (minutos)")
    parser.add_argument("--output-dir", default=None,
                        help="Diretório de saída para os arquivos Markdown gerados")
    parser.add_argument("--extra-args", default="",
                        help="Argumentos extras a serem passados para summarize_transcript.py")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERRO: arquivo não encontrado: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("ERRO: --models precisa conter pelo menos um modelo válido.")
        sys.exit(1)

    print(f"Transcrição: {input_path}")
    print(f"Provider: {args.provider}")
    print(f"Modelos: {', '.join(models)}")
    print(f"Saída: {output_dir}")

    for model in models:
        safe_model = model.replace(':', '_').replace('/', '_')
        output_path = output_dir / f"{input_path.stem}_{safe_model}_summary.md"
        cmd = [sys.executable, "summarize_transcript.py", str(input_path),
               "--provider", args.provider,
               "--model", model,
               "--output", str(output_path)]

        if args.chunk_minutes is not None:
            cmd += ["--chunk-minutes", str(args.chunk_minutes)]
        if args.extra_args:
            cmd += shlex.split(args.extra_args)

        print(f"\n=== Gerando resumo para {model} ===")
        print(" ".join(shlex.quote(part) for part in cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Falha no modelo {model} (código {result.returncode}).")
            sys.exit(result.returncode)
        print(f"Resumo gerado: {output_path}")

    print("\nTodos os resumos foram gerados.")


if __name__ == "__main__":
    main()
