# Changelog

Todas as mudanças notáveis neste projeto estão documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e o projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [Unreleased]

### Adicionado

- `process_video.py`: orquestrador do pipeline completo (extração → resumo → podcast)
  - Executa os 3 passos em sequência com um único comando
  - Barra de progresso geral e temporizador por passo (`fmt_duration`, `progress_bar`, `banner`)
  - Tabela de tempos no final com duração de cada etapa e percentual do total
  - Flags `--skip-summary` e `--skip-podcast` para execução parcial
  - Localiza automaticamente o JSON gerado pelo `extract_transcript.py` via glob com data de hoje

- `generate_podcast.py`: modo de comparação sequencial entre modelos (`--compare-models`)
  - Aceita lista de modelos separados por vírgula: `--compare-models "llama3.1,gemma3:4b,mistral"`
  - Executa geração de roteiro (e opcionalmente áudio) para cada modelo em sequência
  - Chama `keep_alive: 0` na API do Ollama entre modelos para descarregar da RAM antes do próximo
  - Exibe tabela comparativa ao final: falas, palavras totais, média/mín/máx por fala, falas curtas, tempos
  - Gera automaticamente um arquivo Markdown `<video>_comparison.md` com métricas e roteiros por modelo
  - Inclui seções por modelo com falas transcritas para facilitar revisão
  - Funciona com `--script-only` para comparação rápida só dos roteiros

- `script_to_md.py`: utilitário para converter um ou mais roteiros JSON de podcast em Markdown
  - Usa os arquivos `*_script_*.json` já existentes
  - Gera arquivos `<script>_summary.md` automaticamente
  - Permite processamento em lote com múltiplos inputs num único comando

- `summarize_transcript_batch.py`: novo utilitário para gerar resumos executivos em Markdown para vários modelos a partir de uma única transcrição JSON
  - Executa `summarize_transcript.py` para cada modelo listado
  - Gera arquivos `<transcript>_<modelo>_summary.md`
  - Suporte a `--provider`, `--chunk-minutes` e `--output-dir`

- `generate_podcast.py`: prompt de roteiro reescrito com diretrizes de tamanho mínimo por fala
  - Exige mínimo de 100 palavras por fala; descreve regras por papel (host vs. co-host)
  - Passagem de expansão automática (`expand_short_falas`): detecta falas abaixo de 80 palavras e as expande individualmente com contexto da fala anterior/posterior

- `generate_podcast.py`: progresso em tempo real na geração via Ollama
  - Streaming com contador de chars recebidos e tempo decorrido (atualizado por token)
  - Modelo padrão Ollama atualizado para `llama3.1`; `stream=True` substitui `stream=False`

- `generate_podcast.py`: barra de progresso no TTS em batches
  - Exibe `[████░░░░] XX%  N/M segmentos  ⏱ Xs` após cada batch de 5 segmentos

- `summarize_transcript.py`: análise por janela migrada de JSON estruturado para narrativa livre
  - Prompts reescritos para extrair participantes por nome, debates com réplicas/tréplicas e citações
  - Corrige erros de transcrição (ex: "inúrupta" → "ininterrupta") via instrução explícita no prompt
  - Prompt de síntese com nova seção "Debates e Posições" e Linha do Tempo com coluna Interlocutor
  - Metadados injetados diretamente via `{metadata_table}` no template (sem `replace()` frágil)
  - Modelo padrão Ollama atualizado de `llama3.2` para `llama3.1`
  - Chunk padrão Ollama aumentado de 5 min para 10 min (contexto maior para identificar interlocutores)

- `summarize_transcript.py`: barra de progresso por janela de análise
  - Mostra `[████░░] XX%  HH:MM→HH:MM  ETA MM:SS` antes de cada janela
  - Exibe tempo de cada janela após conclusão e resumo final (total + média/janela)
  - Etapa de síntese também reporta tempo de conclusão

### Corrigido

- `extract_transcript.py`: regex de extração de `video_id` agora suporta URLs no formato `/live/` (YouTube Live)
- `generate_podcast.py`: provider Ollama migrado para streaming, eliminando timeouts em roteiros longos

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
