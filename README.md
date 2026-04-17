# YouTubeProcessor

Ferramentas para extrair e processar transcrições de vídeos do YouTube com timestamps.

## Visão Geral

Dois scripts independentes, cada um com uma abordagem diferente:

| Script | Abordagem | Identificação de Falantes |
|---|---|---|
| `extract_transcript.py` | Legendas nativas do YouTube | Não |
| `diarize_transcript.py` | Download de áudio + Whisper + pyannote | Sim |

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

# Escolher diretório de saída
python extract_transcript.py <URL> --output-dir ./meus_outputs
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

## Processamento posterior

Os arquivos JSON gerados são adequados para pipelines de NLP. Cada segmento contém `text`, `start`, `duration` e opcionalmente `speaker`, facilitando:

- Resumo automático (LLMs)
- Análise de sentimento por falante
- Indexação e busca full-text
- Geração de atas de reunião
