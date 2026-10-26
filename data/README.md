# data/

Put **clean speech** recordings for training in `data/clean/`. Nothing in this folder is committed to git.

```
data/clean/
├── speaker_01/          # optional speaker folders -> training/validation split by speaker
│   ├── utt_001.wav
│   └── utt_002.wav
├── speaker_02/
│   └── ...
└── single_file.wav      # files directly in data/clean/ are split by recording
```

* Formats: WAV or FLAC (any sample rate, mono or stereo – they are converted to 16 kHz mono).
* Files must be **clean** (recorded in a quiet room). Noisy versions are generated automatically
  at several SNRs and noise colours by `train_model.py`.
* Recommended: at least 50–100 utterances from 5+ speakers. More data gives a more general model.

## Where to get clean speech

| Dataset | Notes |
|---|---|
| [LibriSpeech](https://www.openslr.org/12) `dev-clean` (≈ 340 MB) | Audiobook speech, FLAC, organised as `speaker/chapter/utt.flac` (works as-is, CC BY 4.0) |
| [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) | 110 speakers, studio quality |
| Your own recordings | 20+ sentences from several classmates, recorded in a quiet room |

For LibriSpeech, extract the archive and point the script to it, e.g.

```bash
python train_model.py --data-dir path/to/LibriSpeech/dev-clean --max-files 300
```

`--max-files`, `--max-seconds` and `--max-frames` keep memory and training time manageable on a laptop.
