# China Japan Classification with ResNet50

Proyek klasifikasi citra antara kelas China dan Japan menggunakan model ResNet50 dengan pipeline data, training, dan evaluasi yang terstruktur.

## Struktur proyek

- `configs/` berisi semua hyperparameter dan konfigurasi global.
- `src/data/` menangani pemuatan dataset, preprocessing, dan split data.
- `src/models/` berisi arsitektur model dan konfigurasi model.
- `src/training/` berisi loop training, callbacks, dan checkpoint.
- `src/evaluation/` berisi metrik, confusion matrix, dan visualisasi.
- `notebooks/` berisi notebook eksplorasi dan baseline training.
- `experiments/` berisi catatan hasil eksperimen.
- `checkpoints/` untuk menyimpan model terbaik.
- `results/` untuk menyimpan visualisasi dan artifact evaluasi.

## Persiapan lingkungan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Jalur data

Siapkan dataset dengan struktur folder seperti:

```text
data/
├── train/
│   ├── china/
│   └── japan/
├── val/
│   ├── china/
│   └── japan/
└── test/
    ├── china/
    └── japan/
```

## Workflow

1. Jalankan notebook inspeksi dataset.
2. Lakukan training pada notebook baseline.
3. Evaluasi performa model dan simpan hasil ke `results/`.
4. Catat eksperimen pada `experiments/README.md`.

## Catatan

Semua parameter training disimpan di `configs/config.py` agar mudah diubah dan dikelola.
