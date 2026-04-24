"""
script_to_md.py

Converte roteiro JSON de podcast (`*_script_*.json`) em Markdown legível.

Uso:
  python script_to_md.py scripts/*.json
  python script_to_md.py transcripts/..._script_llama3_4b.json --output-dir summaries

O arquivo gerado fica no mesmo diretório do JSON por padrão, com sufixo `_summary.md`.
"""

import argparse
import json
from pathlib import Path


def render_script_to_markdown(script: dict, source_name: str) -> str:
    titulo = script.get("titulo", "Sem título")
    descricao = script.get("descricao", "")
    duracao = script.get("duracao_estimada", "—")
    falas = script.get("falas", [])

    lines = []
    lines.append(f"# Roteiro de Podcast — {titulo}")
    lines.append("")
    lines.append(f"**Fonte:** {source_name}")
    lines.append(f"**Duração estimada:** {duracao}")
    if descricao:
        lines.append(f"**Descrição:** {descricao}")
    lines.append("")
    lines.append(f"**Total de falas:** {len(falas)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for index, fala in enumerate(falas, 1):
        speaker = fala.get("speaker", fala.get("host", fala.get("apresentador", "Speaker")))
        texto = fala.get("texto", fala.get("text", fala.get("fala", "")))
        lines.append(f"## Fala {index}: {speaker}")
        lines.append("")
        lines.append(texto.strip())
        lines.append("")

    return "\n".join(lines)


def convert_file(input_path: Path, output_dir: Path | None = None) -> Path:
    with input_path.open("r", encoding="utf-8") as f:
        script = json.load(f)

    output_name = input_path.stem + "_summary.md"
    output_path = (output_dir / output_name) if output_dir else input_path.with_name(output_name)

    markdown = render_script_to_markdown(script, str(input_path.name))
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte um ou mais arquivos JSON de roteiro de podcast em Markdown legível."
    )
    parser.add_argument("inputs", nargs="+", help="Arquivos JSON de roteiro (`*_script_*.json`).")
    parser.add_argument("--output-dir", default=None,
                        help="Diretório onde os arquivos Markdown serão gravados. Padrão: mesmo diretório do JSON.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in args.inputs:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"ERRO: arquivo não encontrado: {input_path}")
            continue
        try:
            out_path = convert_file(input_path, output_dir)
            print(f"Convertido: {input_path} -> {out_path}")
        except Exception as exc:
            print(f"Falha ao converter {input_path}: {exc}")


if __name__ == "__main__":
    main()
