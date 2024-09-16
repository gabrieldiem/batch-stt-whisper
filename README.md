# Batch STT using Whisper from OpenAI

Using STT (Speech to text), transcribe to a text fil all audio files from a folder using local inference with open source OpenAI's Whisper LLM for audio transcription.

## Requirements

An Nvidia graphic card with CUDA capability is required. Information about different Whisper model sizes [here](https://github.com/openai/whisper/blob/main/model-card.md).

The program is set up to use model size `medium` that has 769M parameters and can be loaded with ~5 GB of GPU VRAM. If you want to change the size of the model you can change the `WHISPER_MODEL_ID` constant in the script.

> Tested with Nvidia RTX 3060 with 6GB of VRAM.

## Run the script

Install the requirements:

```shell
pip install -r ./requirements.txt
```

Run the script:

```shell
python ./whisper_process_folder.py path/to/folder/with/audios
```

## Development

### Compile requirements

```shell
pip-compile --output-file=requirements.txt "./requirements.in"
```
