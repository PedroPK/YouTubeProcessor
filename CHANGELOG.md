# Changelog

Todas as mudanças notáveis neste projeto estão documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e o projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [1.0.0] — 2026-04-17

### Adicionado

- `extract_transcript.py`: extração de transcrições via legendas nativas do YouTube
  - Suporte a múltiplos idiomas com ordem de preferência (`--langs`)
  - Fallback automático para qualquer legenda disponível quando o idioma solicitado não existe
  - Comando `--list-langs` para listar idiomas disponíveis antes de extrair
  - Saída nos formatos JSON (com metadados), TXT (leitura humana) e SRT (compatível com players)
  - Timestamps por segmento em todos os formatos de saída
  - Obtenção automática do título do vídeo via `yt-dlp`

- `diarize_transcript.py`: transcrição offline com identificação de falantes
  - Download de áudio via `yt-dlp` (sem download de vídeo)
  - Transcrição com [OpenAI Whisper](https://github.com/openai/whisper) (modelos: tiny, base, small, medium, large, large-v2, large-v3)
  - Diarização de falantes via [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio)
  - Combinação de segmentos Whisper com intervalos de diarização por sobreposição temporal
  - Modo `--no-diarization` para usar apenas Whisper sem necessidade de token HuggingFace
  - Limpeza automática do arquivo de áudio temporário após processamento

- `requirements.txt` com dependências base e comentários sobre dependências opcionais
- `.gitignore` configurado para Python, `.venv/`, `transcripts/` e arquivos de áudio temporários
