"""Adapter for the VocBulwark `vocoder-large` BigVGAN vocoder.

The Hub model takes `(mel_spectrogram, speaker_embedding)` and returns a `.audio` field,
whereas the rest of Matcha calls a vocoder as `vocoder(mel) -> [B, 1, T]`. This wraps the
former in the latter so `matcha.cli.to_waveform` works unchanged.

The speaker embedding is a *vocoder* input, not something the acoustic model predicts: it is
computed from reference audio (see `scripts/compute_speaker_embedding.py`) and held fixed here.
Two layouts are supported, told apart by the tensor's shape:

  * `[768]` or `[1, 768]` -- one embedding, used for every utterance (single-speaker corpus).
  * `[N, 768]`            -- one row per Matcha speaker id, indexed by the `spks` tensor that the
                              acoustic model was trained with (multi-speaker corpus). Row i must
                              be speaker i of the filelist; the script guarantees that.

Two properties of this vocoder are fixed and cannot be configured away: it emits 24 kHz audio,
and it embeds a 50-bit provenance watermark in everything it produces.
"""

import torch

GENERATOR_REPO = "mlr2000/vocoder-large"
SPEAKER_ENCODER_REPO = "mlr2000/vocoder-large-speaker-encoder"

SAMPLE_RATE = 24000  # the vocoder's output rate; not adjustable


class VocBulwarkVocoder(torch.nn.Module):
    """Presents the Hub generator with Matcha's `vocoder(mel, spks=None) -> [B, 1, T]` interface."""

    def __init__(self, generator, speaker_embedding):
        super().__init__()
        self.generator = generator
        # [768] -> [1, 768]; [N, 768] stays. A buffer so .to(device) moves it with the module
        # and it lands in no optimiser.
        self.register_buffer("speaker_embedding", speaker_embedding.reshape(-1, speaker_embedding.shape[-1]))

    @property
    def n_speakers(self):
        return self.speaker_embedding.shape[0]

    def forward(self, mel, spks=None):
        """Vocode `mel` [B, n_mels, T].

        `spks` is the same LongTensor [B] of speaker ids Matcha was conditioned on. It is
        ignored for a single-row table and required for a multi-row one.
        """
        if self.n_speakers == 1:
            emb = self.speaker_embedding.expand(mel.shape[0], -1)
        else:
            if spks is None:
                raise ValueError(
                    f"this VocBulwark vocoder holds {self.n_speakers} speaker embeddings, pass spks (LongTensor [B])"
                )
            spks = torch.as_tensor(spks, device=self.speaker_embedding.device).reshape(-1).long()
            if spks.numel() == 1 and mel.shape[0] > 1:
                spks = spks.expand(mel.shape[0])
            emb = self.speaker_embedding[spks]
        return self.generator(mel_spectrogram=mel, speaker_embedding=emb.to(mel.dtype)).audio

    def remove_weight_norm(self):
        # Matcha calls this on HiFi-GAN; the exported generator already has it folded in.
        pass


def load_speaker_encoder(device="cpu", repo=SPEAKER_ENCODER_REPO):
    from transformers import AutoModel  # pylint: disable=import-outside-toplevel

    return AutoModel.from_pretrained(repo, trust_remote_code=True).eval().to(device)


def load_vocbulwark(speaker_embedding_path, device="cpu", repo=GENERATOR_REPO):
    """Load the generator and pin it to a precomputed speaker embedding or embedding table."""
    from transformers import AutoModel  # pylint: disable=import-outside-toplevel

    generator = AutoModel.from_pretrained(repo, trust_remote_code=True).eval()

    emb = torch.load(speaker_embedding_path, map_location="cpu", weights_only=True)
    if emb.dim() > 2:
        raise ValueError(f"speaker embedding must be [768] or [N, 768], got {tuple(emb.shape)}")
    if emb.shape[-1] != generator.config.speaker_embedding_size:
        raise ValueError(
            f"speaker embedding has {emb.shape[-1]} dims, "
            f"but {repo} expects {generator.config.speaker_embedding_size}"
        )

    return VocBulwarkVocoder(generator, emb).to(device).eval()
