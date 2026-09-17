"""Compute the fixed speaker embedding the VocBulwark vocoder needs.

The vocoder is speaker-conditioned, but LJSpeech is single-speaker, so one embedding is
computed once from reference audio and reused for every utterance. Averaging several clips
suppresses clip-specific variation (recording noise, a particular prosody).

    python scripts/compute_speaker_embedding.py \
        --wav-dir data/LJSpeech-1.1/wavs --n-clips 50 \
        --out data/ljspeech_speaker_embedding.pt

Averaging is the usual approach for speaker embeddings, but the encoder does not document
whether its space is L2-normalised, so `--l2-normalise` is offered and the two are worth an
A/B listen. Use `--n-clips 1` to compare against a single reference clip.
"""

import argparse
from pathlib import Path

import soundfile as sf
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--wav-dir", type=Path, required=True, help="Folder of reference wavs")
    parser.add_argument("--n-clips", type=int, default=50, help="How many clips to average over")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the .pt embedding")
    parser.add_argument("--l2-normalise", action="store_true", help="L2-normalise the averaged embedding")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    import torchaudio.functional as AF  # pylint: disable=import-outside-toplevel

    from matcha.vocbulwark import load_speaker_encoder  # pylint: disable=import-outside-toplevel

    enc = load_speaker_encoder(args.device)
    raw_sr = enc.config.raw_sample_rate

    clips = sorted(args.wav_dir.glob("*.wav"))[: args.n_clips]
    if not clips:
        raise SystemExit(f"no .wav files found under {args.wav_dir}")

    embeddings = []
    for path in clips:
        wav, sr = sf.read(str(path), dtype="float32")
        wav = torch.tensor(wav)
        if wav.dim() == 2:
            wav = wav.mean(1)
        if sr != raw_sr:
            wav = AF.resample(wav, sr, raw_sr)
        with torch.no_grad():
            embeddings.append(enc.embed(wav[None].to(args.device)).cpu())

    stacked = torch.cat(embeddings)
    emb = stacked.mean(0, keepdim=True)
    if args.l2_normalise:
        emb = torch.nn.functional.normalize(emb, dim=-1)

    # Spread across clips is a rough check that the encoder is behaving: near-zero would mean
    # it is ignoring the audio, very large that the clips disagree about the speaker.
    spread = (stacked - stacked.mean(0, keepdim=True)).norm(dim=-1)
    print(f"clips averaged   : {len(clips)} (encoder input {raw_sr} Hz)")
    print(f"embedding        : {tuple(emb.shape)}  norm={emb.norm():.4f}")
    print(f"per-clip spread  : mean={spread.mean():.4f}  max={spread.max():.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, args.out)
    print(f"written          : {args.out}")


if __name__ == "__main__":
    main()
