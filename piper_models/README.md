# GHoSt Terminal - Local Read-Aloud Voices (`piper_models`)

Piper TTS `.onnx` + `.onnx.json` voice pairs live in this folder.

Default voices:
- `silas.onnx` + `silas.onnx.json`
- `kara.onnx` + `kara.onnx.json`

**You don't need to add anything.** The first time you click the speaker button on a reply, GHoSt Terminal downloads the selected voice (about 63 MB each) from `rhasspy/piper-voices` on Hugging Face and saves it here. After that it works offline.

Any other Piper `.onnx` voice you drop into this folder automatically appears in the **Read-Aloud Voice** dropdown inside **[ CONFIG ]**.

## Voice Model Attribution & Licenses

- `silas.onnx` (`rhasspy/piper-voices` [`en_US-john-medium` @ `e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/john/medium/MODEL_CARD), `SHA256: 789c6c875726e627ddee93d51d8727859abe9c091c3d141591f4b83c2072e988`): Trained on [LibriVox](https://librivox.org) recordings (finetuned from `en_US-kristin-medium`) under **Public Domain**.
- `kara.onnx` (`rhasspy/piper-voices` [`en_US-kristin-medium` @ `e7229f7d33`](https://huggingface.co/rhasspy/piper-voices/blob/e7229f7d33/en/en_US/kristin/medium/MODEL_CARD), `SHA256: 5849957f929cbf720c258f8458692d6103fff2f0e3d3b19c8259474bb06a18d4`): Trained from scratch on [LibriVox](https://librivox.org) recordings under **Public Domain**.

See `THIRD_PARTY_NOTICES.md` for full details.
