import os
import sys
from typing import Final
import copy
import whisper as wh
from whisper.model import Whisper as WhisperModel

INPUT_PARAMS: Final[int] = 1
FOLDER_PATH_ARGV_POSITION: Final[int] = 1
EXIT_SUCESS: Final[int] = 0
EXIT_FAILURE: Final[int] = -1

MAX_TRY_ATTEMPTS_FILE_CREATION = 10

WHISPER_MODEL_ID: Final[str] = "medium"
WHISPER_MODEL_DEVICE_CUDA: Final[str] = "cuda"
WHISPER_MODEL_VERBOSE: Final[bool] = True
AVAILABLE_AUDIO_FILE_EXTENSIONS: Final[list] = [".m4a", ".mp4", ".mp3"]


def loadWhisperModel() -> WhisperModel:
    print(f"Loading Whisper model with id '{WHISPER_MODEL_ID}'")
    return wh.load_model(
        name=WHISPER_MODEL_ID, device=WHISPER_MODEL_DEVICE_CUDA, in_memory=True
    )


def isAnAudioFile(filename: str):
    for extension in AVAILABLE_AUDIO_FILE_EXTENSIONS:
        if filename.endswith(extension):
            return True
    return False


def readFilesFromFolder(folderPath: str):
    allFilepaths = []
    for filename in os.listdir(folderPath):
        filepath = os.path.join(folderPath, filename)
        if isAnAudioFile(filename):
            allFilepaths.append({"filepath": filepath, "filename": filename})
    return allFilepaths


def saveToFile(text: str, filename: str, folderPath: str):
    newFileName = copy.deepcopy(filename)
    i = 0
    while (
        os.path.exists(f"{folderPath}/{newFileName}.txt")
        and i < MAX_TRY_ATTEMPTS_FILE_CREATION
    ):
        newFileName = f"{newFileName}_new"
        i += 1

    if i == MAX_TRY_ATTEMPTS_FILE_CREATION:
        raise Exception("Cannot create file to save text")

    filepath = f"{folderPath}/{newFileName}.txt"
    with open(filepath, "w") as file:
        file.write(text)
        print(f"Saved transcription to {filepath}\n")


def transcribeFiles(allFilepaths: list, folderPath: str, whisperModel: WhisperModel):
    fileCount = len(allFilepaths)
    i = 1
    for path in allFilepaths:
        try:
            filename = path["filename"]
            filepath = path["filepath"]
            print(f"Transcribing file {i}/{fileCount} with name '{filename}'")

            result = whisperModel.transcribe(
                audio=filepath, verbose=WHISPER_MODEL_VERBOSE
            )
            resultText = result["text"]
            fileNameWithoutExtension = filename.split(".")[0]
            saveToFile(resultText, fileNameWithoutExtension, folderPath)

            i += 1
        except Exception as exception:
            print(f"Error: {exception}")


if __name__ == "__main__":
    try:
        argc = len(sys.argv)

        if argc != INPUT_PARAMS + 1:
            print(f"Error while calling program. Expected: {sys.argv[0]} <folder path>")
            sys.exit(EXIT_FAILURE)

        whisperModel = loadWhisperModel()

        folderPath = sys.argv[FOLDER_PATH_ARGV_POSITION]
        allFilepaths = readFilesFromFolder(folderPath)
        transcribeFiles(allFilepaths, folderPath, whisperModel)

    except Exception as exception:
        print(f"Error: {exception}")
        sys.exit(EXIT_FAILURE)
