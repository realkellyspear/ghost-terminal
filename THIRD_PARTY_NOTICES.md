# Third-Party Notices

GHoSt Terminal is licensed under **GPL-3.0-or-later** (see `LICENSE`). It uses the following third-party components, each under its own license.

## Piper TTS & eSpeak NG (`piper-tts` 1.8.0)
- Project: https://github.com/OHF-Voice/piper1-gpl and https://github.com/espeak-ng/espeak-ng
- License: **GPL-3.0-or-later**

## Piper Voice Models (downloaded on first use into `piper_models/`)
From [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices), snapshot [`e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/tree/e7229f7d33):

- **`silas`** = `en_US-john-medium` (SHA256 `789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988`). Trained on Public Domain LibriVox recordings. License: **Public Domain**. Model card: `piper_models/silas.MODEL_CARD`.
- **`kara`** = `en_US-kristin-medium` (SHA256 `5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4`). Trained on Public Domain LibriVox recordings. License: **Public Domain**. Model card: `piper_models/kara.MODEL_CARD`.

## Runtime libraries
- `llama.cpp` / `llama-cpp-python`: Georgi Gerganov & contributors / Andrei Betlen (**MIT**)
- ONNX Runtime: Microsoft Corporation (**MIT**)
- NVIDIA CUDA runtime (`cublas`): NVIDIA Corporation (redistributable under the NVIDIA CUDA Toolkit EULA)
- FastAPI, Uvicorn, Pydantic, PyWebView, SoundDevice, SoundFile (`libsndfile`, LGPL-2.1): their respective open-source licenses (MIT / BSD / Apache-2.0 / LGPL-2.1)
