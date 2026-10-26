# models/

Trained model files are **generated locally** and are not committed to git.

| File | Created by | Contents |
|---|---|---|
| `speech_enhancer.joblib` | `train_model.py` | Random Forest model + the STFT/feature configuration it was trained with + metadata |
| `training_summary.json` | `train_model.py` | Data source, split, hyper-parameters and held-out validation results |

Create them with one of:

```bash
python train_model.py --data-dir data/clean   # real clean speech (recommended)
python train_model.py --synthetic             # synthetic signals – pipeline test only
```

The dashboard starts without a model; ML enhancement is simply disabled until one exists.

**Security:** joblib files are Python pickles and can run code when loaded. Only load model files
you trained yourself. The app only loads `models/speech_enhancer.joblib` and never loads uploaded files as models.
