import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse

# Upper bound for the Whisper front-end's fixed analysis window. Any value past the longest
# utterance in the corpus works; it only has to prevent truncation.
MAX_AUDIO_SECONDS = 30.0


def load_audio(filepath):
    """Read a waveform as (channels, samples) float32 in [-1, 1], plus its sample rate.

    This used to be `torchaudio.load`, but torchaudio 2.9 dropped its own decoders and now
    routes that call through TorchCodec, which needs system FFmpeg. We cannot simply pin
    torchaudio back: the ROCm wheel index publishes exactly one version (2.11.0), so the AMD
    and NVIDIA builds sit on different sides of that change. Reading via soundfile keeps the
    two identical and drops the FFmpeg requirement entirely.

    Verified bit-identical to `torchaudio.load` on LJSpeech (same shape, dtype and samples).
    """
    audio, sr = sf.read(filepath, dtype="float32", always_2d=True)  # (samples, channels)
    return torch.from_numpy(np.ascontiguousarray(audio.T)), sr


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        mel_frontend="hifigan",
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.mel_frontend,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.mel_frontend,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks),
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
        mel_frontend="hifigan",
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations
        self.mel_frontend = mel_frontend
        # Built lazily: the extractor is created inside each dataloader worker rather than
        # pickled into it.
        self._whisper_fe = None

        if mel_frontend not in ("hifigan", "whisper"):
            raise ValueError(f"unknown mel_frontend {mel_frontend!r}, expected 'hifigan' or 'whisper'")

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, filepath_and_text):
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None

        text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        durations = self.get_durations(filepath, text) if self.load_durations else None

        return {"x": text, "y": mel, "spk": spk, "filepath": filepath, "x_text": cleaned_text, "durations": durations}

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def whisper_extractor(self):
        # The VocBulwark vocoder reads a Whisper-style log-mel, not the HiFi-GAN one, so when
        # targeting it we must reproduce that front-end exactly. `f_min`/`f_max` play no part:
        # Whisper's filterbank always spans 0..sample_rate/2.
        if self._whisper_fe is None:
            from transformers import WhisperFeatureExtractor  # pylint: disable=import-outside-toplevel

            fe = WhisperFeatureExtractor(
                sampling_rate=self.sample_rate,
                n_fft=self.n_fft,
                feature_size=self.n_mels,
                hop_length=self.hop_length,
            )
            # By default the extractor pads or truncates everything to a fixed 30 s window,
            # which would corrupt every utterance. Widen it past the longest clip instead.
            fe.n_samples = int(MAX_AUDIO_SECONDS * self.sample_rate)
            fe.chunk_length = MAX_AUDIO_SECONDS
            self._whisper_fe = fe
        return self._whisper_fe

    def get_whisper_mel(self, audio, sr):
        import torchaudio.functional as AF  # pylint: disable=import-outside-toplevel

        wav = audio.mean(0)                                   # (channels, samples) -> mono
        if sr != self.sample_rate:
            wav = AF.resample(wav, sr, self.sample_rate)      # LJSpeech is 22.05 kHz, target 24 kHz

        # Whisper clamps the noise floor at (peak of *this input* - 8), so the result depends on
        # what it is handed. Always pass the whole utterance and crop the resulting tensor later
        # -- cropping the audio first shifts every frame.
        feats = self.whisper_extractor()(
            wav.numpy(), sampling_rate=self.sample_rate, padding="longest", return_tensors="pt"
        )
        return feats["input_features"].squeeze(0)

    def get_mel(self, filepath):
        audio, sr = load_audio(filepath)

        if self.mel_frontend == "whisper":
            mel = self.get_whisper_mel(audio, sr)
        else:
            assert sr == self.sample_rate
            mel = mel_spectrogram(
                audio,
                self.n_fft,
                self.n_mels,
                self.sample_rate,
                self.hop_length,
                self.win_length,
                self.f_min,
                self.f_max,
                center=False,
            ).squeeze()

        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, add_blank=True):
        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks):
        self.n_spks = n_spks

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }
