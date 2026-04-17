# YouTubeProcessor

Ferramentas para extrair e processar transcrições de vídeos do YouTube com timestamps.

## Visão Geral

Três scripts independentes, organizados em camadas:

| Script | Função | Depende de |
|---|---|---|
| `extract_transcript.py` | Extrai transcrição via legendas nativas do YouTube | `youtube-transcript-api`, `yt-dlp` |
| `diarize_transcript.py` | Baixa áudio e transcreve com identificação de falantes | `whisper`, `pyannote.audio`, `torch` |
| `summarize_transcript.py` | Gera resumo estruturado em Markdown via LLM | `openai` / `anthropic` / Ollama local |
| `generate_podcast.py` | Converte um resumo Markdown em áudio estilo podcast | `edge-tts`, `imageio-ffmpeg`, `audioop-lts` |

---

## Instalação

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para a abordagem com diarização, instale as dependências adicionais:

```bash
pip install openai-whisper pyannote.audio torch torchaudio
```

---

## `extract_transcript.py` — Transcrição via legendas nativas

Usa a API pública do YouTube para obter as legendas (manuais ou auto-geradas) sem baixar o vídeo.

**Prós:** rápido (segundos), sem dependências pesadas.  
**Contras:** não identifica falantes; requer que o vídeo tenha legenda disponível.

Suporta todos os formatos de URL do YouTube: vídeos normais (`watch?v=`), encurtadas (`youtu.be/`), lives (`/live/`) e Shorts (`/shorts/`).

### Uso

```bash
# Listar idiomas disponíveis para um vídeo
python extract_transcript.py <URL> --list-langs

# Extrair com idioma preferido
python extract_transcript.py <URL> --lang pt

# Tentar múltiplos idiomas em ordem de preferência
python extract_transcript.py <URL> --langs pt pt-BR en

# Escolher formatos de saída (padrão: json txt srt)
python extract_transcript.py <URL> --lang pt --formats json txt

# Diretório de saída personalizado
python extract_transcript.py <URL> --output-dir ./meus_outputs

# Funciona também com URLs de live
python extract_transcript.py "https://www.youtube.com/live/ID" --lang pt
```

### Argumentos

| Argumento | Descrição | Padrão |
|---|---|---|
| `url` | URL do vídeo no YouTube | — |
| `--lang` | Idioma preferido (ex: `pt`, `en`) | `pt pt-BR en` |
| `--langs` | Lista de idiomas em ordem de preferência | — |
| `--list-langs` | Lista idiomas disponíveis e encerra | — |
| `--output-dir` | Diretório de saída | `./transcripts` |
| `--formats` | Formatos de saída: `json`, `txt`, `srt` | todos |

### Formatos de saída

**TXT** — leitura humana com timestamps:
```
[00:00:18,640] Não somos estranhos ao amor
[00:00:22,640] Você conhece as regras, e eu também
```

**SRT** — compatível com players de vídeo e ferramentas de legenda:
```
1
00:00:18,640 --> 00:00:21,880
Não somos estranhos ao amor
```

**JSON** — estruturado para processamento programático:
```json
{
  "meta": {
    "video_id": "dQw4w9WgXcQ",
    "title": "...",
    "language": "pt-BR",
    "segment_count": 60,
    "duration_seconds": 211.32
  },
  "segments": [
    { "text": "Não somos estranhos ao amor", "start": 18.64, "duration": 3.24 }
  ]
}
```

---

## `diarize_transcript.py` — Transcrição com identificação de falantes

Baixa o áudio com `yt-dlp`, transcreve com [OpenAI Whisper](https://github.com/openai/whisper) e identifica os falantes com [pyannote.audio](https://github.com/pyannote/pyannote-audio).

**Prós:** funciona mesmo sem legenda; identifica quem fala em cada trecho.  
**Contras:** lento em CPU; requer token gratuito do HuggingFace.

### Pré-requisitos adicionais

1. Aceitar os termos de uso dos modelos no HuggingFace:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

2. Gerar um token de acesso: https://hf.co/settings/tokens

3. Configurar o token:
   ```bash
   export HF_TOKEN="hf_xxxxxxxxxxxx"
   ```

### Uso

```bash
# Transcrição com diarização (token via variável de ambiente)
export HF_TOKEN="hf_xxxx"
python diarize_transcript.py <URL>

# Especificar modelo Whisper e diretório de saída
python diarize_transcript.py <URL> --model large-v2 --output-dir ./transcripts

# Apenas transcrição Whisper, sem diarização
python diarize_transcript.py <URL> --no-diarization

# Passar token diretamente
python diarize_transcript.py <URL> --hf-token hf_xxxx
```

### Modelos Whisper

| Modelo | Tamanho | Velocidade (CPU) | Qualidade |
|---|---|---|---|
| `tiny` | 39 MB | Muito rápido | Baixa |
| `base` | 74 MB | Rápido | Razoável |
| `small` | 244 MB | Moderado | Boa |
| `medium` | 769 MB | Lento | Muito boa |
| `large-v2` | 1.5 GB | Muito lento | Excelente |

> Recomendação para português: `medium` ou `large-v2`.

### Saída com diarização (TXT)

```
[SPEAKER_00]
  [00:00:05,000] Bom dia, vamos começar a reunião.
  [00:00:08,200] Hoje temos três pontos na pauta.

[SPEAKER_01]
  [00:00:12,500] Certo, pode começar.
```

Os rótulos `SPEAKER_00`, `SPEAKER_01`... identificam falantes distintos. O mapeamento para nomes reais deve ser feito manualmente após a extração.

---

---

## `summarize_transcript.py` — Resumo estruturado em Markdown via LLM

Recebe o JSON gerado por `extract_transcript.py` ou `diarize_transcript.py` e produz um documento Markdown com resumo executivo, temas, discussões, decisões, participantes e linha do tempo.

**Estratégia map-reduce:** divide a transcrição em janelas de tempo, resume cada uma individualmente e sintetiza tudo em um documento final — permitindo processar transcrições longas que não caberiam em um único prompt.

### Provedores suportados

| Provider | Modelo padrão | Requisito |
|---|---|---|
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` |
| `ollama` | `llama3.2` | Ollama rodando localmente (gratuito) |

### Uso

```bash
# Com OpenAI (padrão)
export OPENAI_API_KEY="sk-..."
python summarize_transcript.py transcripts/video.json

# Com Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
python summarize_transcript.py transcripts/video.json --provider anthropic

# Com Ollama local (gratuito, sem API key)
ollama pull llama3.2
python summarize_transcript.py transcripts/video.json --provider ollama --model llama3.2

# Ajustar tamanho das janelas de análise (padrão: 15 min)
python summarize_transcript.py transcripts/video.json --chunk-minutes 20

# Especificar arquivo de saída
python summarize_transcript.py transcripts/video.json --output resumo.md
```

### Argumentos

| Argumento | Descrição | Padrão |
|---|---|---|
| `input` | Arquivo JSON da transcrição | — |
| `--provider` | Provedor LLM: `openai`, `anthropic`, `ollama` | `openai` |
| `--model` | Modelo a usar | varia por provider |
| `--chunk-minutes` | Duração de cada janela de análise (min) | `15` |
| `--output` | Arquivo `.md` de saída | mesmo nome do input |

### Saídas geradas

- `<video>.md` — documento Markdown estruturado com todas as seções
- `<video>_chunks.json` — resumos parciais por janela de tempo (para reuso)

---

---

## `generate_podcast.py` — Geração de podcast em áudio a partir do resumo

Converte um arquivo Markdown (gerado por `summarize_transcript.py` ou escrito manualmente) em um episódio de podcast em MP3, com dois apresentadores e vozes sintetizadas em português.

**Pipeline:**
1. LLM transforma o resumo em um roteiro conversacional com dois hosts (JSON)
2. TTS gera um arquivo de áudio por fala (Microsoft Edge TTS — gratuito)
3. `ffmpeg` concatena os segmentos com pausas e exporta o MP3 final

**Pré-requisitos:**
```bash
pip install edge-tts imageio-ffmpeg audioop-lts
```

> `imageio-ffmpeg` inclui um binário pré-compilado do ffmpeg para todas as plataformas — sem necessidade de `brew install ffmpeg`. Em Python 3.13+, `audioop-lts` substitui o módulo `audioop` removido da biblioteca padrão.

### Provedores suportados

| Provider TTS | Qualidade | Custo | API Key |
|---|---|---|---|
| `edge` (padrão) | Muito boa | Gratuito | Não precisa |
| `openai` | Excelente | Pago | `OPENAI_API_KEY` |

**Vozes padrão Edge TTS (pt-BR):** `pt-BR-AntonioNeural` (host) e `pt-BR-FranciscaNeural` (co-host)

### Uso

```bash
# Uso básico (LLM: openai, TTS: edge)
export OPENAI_API_KEY="sk-..."
python generate_podcast.py transcripts/video.md

# Com LLM Anthropic e TTS gratuito
export ANTHROPIC_API_KEY="sk-ant-..."
python generate_podcast.py transcripts/video.md --provider anthropic

# Escolher vozes diferentes
python generate_podcast.py transcripts/video.md --voice1 pt-BR-ThalitaNeural --voice2 pt-BR-FranciscaNeural

# Customizar nomes e duração alvo do podcast
python generate_podcast.py transcripts/video.md --host1 Carlos --host2 Marina --duration-min 10 --duration-max 15

# Apenas gerar o roteiro em JSON (sem áudio)
python generate_podcast.py transcripts/video.md --script-only

# Usar roteiro já gerado (pula o LLM)
python generate_podcast.py transcripts/video.md --from-script transcripts/video_podcast_script.json

# Listar vozes Edge disponíveis para português
python generate_podcast.py transcripts/video.md --list-voices
```

### Argumentos

| Argumento | Descrição | Padrão |
|---|---|---|
| `input` | Arquivo `.md` com o resumo | — |
| `--provider` | Provider LLM para o roteiro: `openai`, `anthropic`, `ollama` | `openai` |
| `--model` | Modelo LLM | varia por provider |
| `--tts` | Provider TTS: `edge`, `openai` | `edge` |
| `--voice1` | Voz do host principal (Edge ou OpenAI) | `pt-BR-AntonioNeural` |
| `--voice2` | Voz do co-host (Edge ou OpenAI) | `pt-BR-FranciscaNeural` |
| `--host1` | Nome do host principal | `Rafael` |
| `--host2` | Nome do co-host | `Ana` |
| `--duration-min` | Duração mínima alvo (min) | `8` |
| `--duration-max` | Duração máxima alvo (min) | `12` |
| `--pause-ms` | Pausa entre falas em ms | `400` |
| `--output` | Arquivo MP3 de saída | mesmo nome do input |
| `--script-only` | Gera apenas o roteiro JSON | — |
| `--from-script` | Usa JSON existente, pula LLM | — |
| `--list-voices` | Lista vozes Edge para pt e encerra | — |

### Saídas geradas

- `<video>.mp3` — episódio de podcast completo
- `<video>_podcast_script.json` — roteiro estruturado com todas as falas

---

## Processamento posterior

Os arquivos JSON gerados são adequados para pipelines de NLP. Cada segmento contém `text`, `start`, `duration` e opcionalmente `speaker`, facilitando:

- Resumo automático com `summarize_transcript.py`
- Geração de episódio de podcast com `generate_podcast.py`
- Análise de sentimento por falante
- Indexação e busca full-text
- Geração de atas de reunião
