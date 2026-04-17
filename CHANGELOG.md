# Changelog

Todas as mudanças notáveis neste projeto estão documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e o projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [Unreleased]

### Adicionado

- `generate_podcast.py`: converte resumo Markdown em episódio de podcast em MP3
  - LLM gera roteiro conversacional com dois hosts em português
  - TTS Microsoft Edge (gratuito) com vozes `pt-BR-AntonioNeural` e `pt-BR-FranciscaNeural`
  - Suporte a TTS OpenAI como alternativa paga
  - `ffmpeg` (via `imageio-ffmpeg` bundled) concatena segmentos com pausas configuráveis e exporta MP3
  - `imageio-ffmpeg` fornece binário pré-compilado do ffmpeg — sem necessidade de instalação via brew/apt
  - `audioop-lts` garante compatibilidade com Python 3.13+ (módulo `audioop` removido da stdlib)
  - Modo `--script-only` para gerar apenas o roteiro JSON sem áudio
  - Modo `--from-script` para reutilizar roteiro já gerado
  - Argumento `--list-voices` lista todas as vozes Edge TTS para português

- `summarize_transcript.py`: geração de resumo estruturado em Markdown via LLM
  - Estratégia map-reduce: divide a transcrição em janelas de tempo, resume cada janela e sintetiza o documento final
  - Suporte a três provedores: OpenAI (`gpt-4o-mini`), Anthropic (`claude-3-5-haiku`) e Ollama local (`llama3.2`)
  - Documento Markdown gerado com: resumo executivo, temas, discussões, decisões, participantes e linha do tempo
  - Salva resumos parciais por janela em `_chunks.json` para reuso sem reprocessar
  - Argumento `--chunk-minutes` para controlar o tamanho das janelas de análise (padrão: 15 min)

### Corrigido

- `extract_transcript.py`: regex de extração de `video_id` agora suporta URLs no formato `/live/` (YouTube Live)

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
  - Suporte a URLs: `watch?v=`, `youtu.be/`, `/embed/`, `/shorts/`

- `diarize_transcript.py`: transcrição offline com identificação de falantes
  - Download de áudio via `yt-dlp` (sem download de vídeo)
  - Transcrição com [OpenAI Whisper](https://github.com/openai/whisper) (modelos: tiny, base, small, medium, large, large-v2, large-v3)
  - Diarização de falantes via [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio)
  - Combinação de segmentos Whisper com intervalos de diarização por sobreposição temporal
  - Modo `--no-diarization` para usar apenas Whisper sem necessidade de token HuggingFace
  - Limpeza automática do arquivo de áudio temporário após processamento

- `requirements.txt` com dependências base e comentários sobre dependências opcionais
- `.gitignore` configurado para Python, `.venv/`, `transcripts/` e arquivos de áudio temporários
