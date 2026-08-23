from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Tuple


# ============================================================
# Sub-configs (satu dataclass per domain)
# ============================================================

@dataclass
class DataConfig:
    """
    Semua parameter yang berkaitan dengan dataset dan input pipeline.

    train_ratio + val_ratio + test_ratio harus = 1.0
    Split dilakukan secara stratified agar distribusi kelas seimbang.
    """
    image_size: Tuple[int, int] = (224, 224)   # ResNet50 standard input
    batch_size: int = 32
    num_classes: int = 2
    class_names: List[str] = field(
        default_factory=lambda: ["chinese", "japanese"]
    )
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    use_augmentation: bool = False  # False untuk baseline murni


@dataclass
class ModelConfig:
    """
    Arsitektur model.

    Classification head yang digunakan:
        ResNet50 (pretrained)
            ↓
        GlobalAveragePooling2D
            ↓
        Dropout(dropout_rate)
            ↓
        Dense(1, activation='sigmoid')   ← binary classification

    Kenapa sigmoid + Dense(1)?
    Karena ini binary classification (Chinese vs Japanese).
    Alternatif softmax + Dense(2) menghasilkan hasil identik,
    tapi sigmoid + Dense(1) lebih umum untuk kasus binary dan
    lebih efisien secara komputasi.
    """
    backbone: str = "resnet50"
    pretrained_weights: str = "imagenet"
    dropout_rate: float = 0.5
    # Catatan: classification head didefinisikan di src/models/resnet50.py


@dataclass
class TrainingConfig:
    """
    Hyperparameter training untuk kedua stage.

    Stage 1 — Frozen backbone:
        Hanya classification head yang dilatih.
        LR bisa lebih tinggi karena backbone tidak berubah.

    Stage 2 — Fine-tuning:
        Backbone sebagian/seluruhnya di-unfreeze.
        LR sangat kecil untuk menghindari catastrophic forgetting.

    fine_tune_from_layer:
        -1  → tidak ada fine-tuning (Exp A: frozen saja)
        143 → unfreeze dari layer 143+ (Exp B: partial, conv5_block)
        0   → unfreeze semua layer (Exp C: full fine-tuning)

    Layer count ResNet50 tanpa top ≈ 175 layer.
    Layer 143 kira-kira awal dari conv5_block1 (last residual group).
    """
    # Stage 1
    stage1_epochs: int = 10
    stage1_lr: float = 1e-3

    # Stage 2
    stage2_epochs: int = 20
    stage2_lr: float = 1e-5
    fine_tune_from_layer: int = -1   # -1 = frozen (stage 1 only)

    # Callbacks
    early_stopping_patience: int = 7
    reduce_lr_patience: int = 3
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7

    random_seed: int = 42


@dataclass
class PathConfig:
    """
    File system paths.
    dataset_dir HARUS di-set sebelum cfg.validate() dipanggil.

    Gunakan @property untuk derived paths agar tidak perlu
    update manual ketika experiment_name berubah.
    """
    dataset_dir: str = ""
    results_dir: str = "results"
    checkpoints_dir: str = "checkpoints"
    experiment_name: str = "experiment"

    @property
    def experiment_results_dir(self) -> str:
        """Folder untuk menyimpan hasil eksperimen ini."""
        return os.path.join(self.results_dir, self.experiment_name)

    @property
    def experiment_checkpoint_dir(self) -> str:
        """Folder untuk menyimpan checkpoint model eksperimen ini."""
        return os.path.join(self.checkpoints_dir, self.experiment_name)


# ============================================================
# Root config
# ============================================================

@dataclass
class Config:
    """
    Root configuration object.
    Semua sub-config di-compose di sini.

    Contoh penggunaan:
        cfg = get_baseline_config(dataset_dir="/content/dataset")
        cfg.validate()
        print(cfg.training.stage1_lr)   # 0.001
        print(cfg.paths.experiment_results_dir)  # results/exp_a_frozen
    """
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    def validate(self) -> None:
        """
        Sanity check sebelum eksperimen dimulai.
        Panggil ini di awal notebook setelah config di-set.
        Akan raise AssertionError dengan pesan jelas jika ada yang salah.
        """
        assert self.paths.dataset_dir, (
            "paths.dataset_dir harus di-set. "
            "Contoh: cfg.paths.dataset_dir = '/content/dataset'"
        )
        assert os.path.exists(self.paths.dataset_dir), (
            f"dataset_dir tidak ditemukan: {self.paths.dataset_dir}"
        )

        split_sum = (
            self.data.train_ratio
            + self.data.val_ratio
            + self.data.test_ratio
        )
        assert abs(split_sum - 1.0) < 1e-6, (
            f"train + val + test harus = 1.0, tapi = {split_sum:.4f}"
        )

        assert self.data.num_classes == len(self.data.class_names), (
            f"num_classes ({self.data.num_classes}) harus sama dengan "
            f"panjang class_names ({len(self.data.class_names)})"
        )

        assert 0.0 < self.data.train_ratio < 1.0, "train_ratio harus antara 0 dan 1"
        assert self.training.stage1_lr > 0, "stage1_lr harus positif"
        assert self.training.stage2_lr > 0, "stage2_lr harus positif"
        assert self.training.stage2_lr < self.training.stage1_lr, (
            "stage2_lr (fine-tuning) harus lebih kecil dari stage1_lr"
        )

        print("✅ Config valid.")

    def save(self, path: str) -> None:
        """
        Simpan config ke JSON untuk reproducibility.
        Dipanggil otomatis oleh logger.save_experiment().

        PathConfig menggunakan @property yang tidak di-handle asdict,
        sehingga kita serialize secara manual.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "paths": {
                "dataset_dir": self.paths.dataset_dir,
                "results_dir": self.paths.results_dir,
                "checkpoints_dir": self.paths.checkpoints_dir,
                "experiment_name": self.paths.experiment_name,
            },
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Config disimpan → {path}")

    @classmethod
    def load(cls, path: str) -> Config:
        """
        Rekonstruksi Config dari file JSON yang sudah disimpan.
        Berguna untuk reproduce eksperimen lama.

        Contoh:
            cfg = Config.load("results/exp_a_frozen/config.json")
        """
        with open(path) as f:
            data = json.load(f)
        cfg = cls()
        cfg.data = DataConfig(**data["data"])
        cfg.model = ModelConfig(**data["model"])
        cfg.training = TrainingConfig(**data["training"])
        cfg.paths = PathConfig(**data["paths"])
        return cfg

    def __repr__(self) -> str:
        lines = [
            f"Config(experiment='{self.paths.experiment_name}')",
            f"  data.image_size        : {self.data.image_size}",
            f"  data.batch_size        : {self.data.batch_size}",
            f"  data.class_names       : {self.data.class_names}",
            f"  data.split             : {self.data.train_ratio}/{self.data.val_ratio}/{self.data.test_ratio}",
            f"  model.backbone         : {self.model.backbone}",
            f"  model.dropout          : {self.model.dropout_rate}",
            f"  training.stage1_epochs : {self.training.stage1_epochs}",
            f"  training.stage1_lr     : {self.training.stage1_lr}",
            f"  training.fine_tune_from: {self.training.fine_tune_from_layer}",
            f"  training.stage2_epochs : {self.training.stage2_epochs}",
            f"  training.stage2_lr     : {self.training.stage2_lr}",
            f"  paths.dataset_dir      : {self.paths.dataset_dir}",
            f"  paths.results          : {self.paths.experiment_results_dir}",
        ]
        return "\n".join(lines)


# ============================================================
# Factory functions — satu per eksperimen
# ============================================================

def get_baseline_config(dataset_dir: str) -> Config:
    """
    Experiment A — Frozen backbone.

    ResNet50 ImageNet weights sepenuhnya dibekukan.
    Hanya classification head (GAP → Dropout → Dense) yang dilatih.
    Ini adalah eksperimen paling konservatif dan paling cepat dijalankan.
    Hasilnya mencerminkan kualitas ImageNet features tanpa adaptasi domain.
    """
    cfg = Config()
    cfg.paths.dataset_dir = dataset_dir
    cfg.paths.experiment_name = "exp_a_frozen"
    cfg.training.fine_tune_from_layer = -1   # backbone frozen
    return cfg


def get_partial_finetune_config(dataset_dir: str) -> Config:
    """
    Experiment B — Partial fine-tuning (conv5_block).

    Unfreeze layer 143 ke atas (≈ last residual block ResNet50).
    Stage 1 melatih head dulu (sama seperti Exp A),
    kemudian Stage 2 membuka sebagian backbone dengan LR kecil.
    Idealnya dimulai dari checkpoint terbaik Exp A.
    """
    cfg = Config()
    cfg.paths.dataset_dir = dataset_dir
    cfg.paths.experiment_name = "exp_b_partial_finetune"
    cfg.training.fine_tune_from_layer = 143  # conv5_block1 dan seterusnya
    return cfg


def get_full_finetune_config(dataset_dir: str) -> Config:
    """
    Experiment C — Full fine-tuning.

    Seluruh backbone dapat di-update.
    Risiko overfitting paling tinggi jika dataset kecil.
    Gunakan EarlyStopping dan pantau val_loss dengan ketat.
    Idealnya dimulai dari checkpoint terbaik Exp B.
    """
    cfg = Config()
    cfg.paths.dataset_dir = dataset_dir
    cfg.paths.experiment_name = "exp_c_full_finetune"
    cfg.training.fine_tune_from_layer = 0    # semua layer trainable
    return cfg