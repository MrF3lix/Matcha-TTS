"""Compute the fixed speaker embedding(s) the VocBulwark vocoder needs.

The vocoder is speaker-conditioned, so a 768-d embedding is computed from reference audio and
reused at synthesis time. Averaging several clips suppresses clip-specific variation (recording
noise, a particular prosody).

Single-speaker corpus -- one embedding from a folder of wavs:

    python scripts/compute_speaker_embedding.py \
        --wav-dir data/LJSpeech-1.1/wavs --n-clips 50 \
        --out data/ljspeech_speaker_embedding.pt

Multi-speaker corpus -- one embedding per speaker, driven by a Matcha filelist
(`path|speaker_id|text`, as written by scripts/build_filelist.py). The result is an [N, 768]
table whose row i is speaker id i, so Matcha's `spks` tensor indexes it directly:

    python scripts/compute_speaker_embedding.py \
        --filelist data/swissgerman/filelists/train.txt --n-clips 50 \
        --out data/swissgerman/speaker_embeddings.pt

Averaging is the usual approach for speaker embeddings, but the encoder does not document
whether its space is L2-normalised, so `--l2-normalise` is offered and the two are worth an
A/B listen. Use `--n-clips 1` to compare against a single reference clip.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import soundfile as sf
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav-dir", type=Path, help="Folder of reference wavs (single speaker)")
    src.add_argument("--filelist", type=Path, help="Matcha filelist `path|speaker_id|text` (one embedding per speaker)")
    parser.add_argument("--n-clips", type=int, default=50, help="How many clips to average over (per speaker)")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the .pt embedding / table")
    parser.add_argument("--l2-normalise", action="store_true", help="L2-normalise the averaged embedding(s)")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def clips_per_speaker(args):
    """Return {speaker_id: [wav paths]} with ids 0..N-1; a single speaker gets id 0."""
    if args.wav_dir is not None:
        clips = sorted(args.wav_dir.glob("*.wav"))
        if not clips:
            raise SystemExit(f"no .wav files found under {args.wav_dir}")
        return {0: clips[: args.n_clips]}

    groups = defaultdict(list)
    with open(args.filelist, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 3:
                raise SystemExit(
                    f"{args.filelist}: expected `path|speaker_id|text`, got {len(parts)} fields: {line.strip()[:80]}"
                )
            groups[int(parts[1])].append(Path(parts[0]))

    ids = sorted(groups)
    if ids != list(range(len(ids))):
        raise SystemExit(f"speaker ids must be contiguous 0..N-1 so they can index the table, got {ids}")
    return {spk: sorted(paths)[: args.n_clips] for spk, paths in groups.items()}


def embed_clips(enc, clips, raw_sr, device):
    import torchaudio.functional as AF  # pylint: disable=import-outside-toplevel

    embeddings = []
    for path in clips:
        wav, sr = sf.read(str(path), dtype="float32")
        wav = torch.tensor(wav)
        if wav.dim() == 2:
            wav = wav.mean(1)
        if sr != raw_sr:
            wav = AF.resample(wav, sr, raw_sr)
        with torch.no_grad():
            embeddings.append(enc.embed(wav[None].to(device)).cpu())
    return torch.cat(embeddings)


def main():
    args = parse_args()

    from matcha.vocbulwark import load_speaker_encoder  # pylint: disable=import-outside-toplevel

    enc = load_speaker_encoder(args.device)
    raw_sr = enc.config.raw_sample_rate

    groups = clips_per_speaker(args)
    rows = []
    for spk in sorted(groups):
        stacked = embed_clips(enc, groups[spk], raw_sr, args.device)
        emb = stacked.mean(0, keepdim=True)
        if args.l2_normalise:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        rows.append(emb)

        # Spread across clips is a rough check that the encoder is behaving: near-zero would
        # mean it is ignoring the audio, very large that the clips disagree about the speaker.
        spread = (stacked - stacked.mean(0, keepdim=True)).norm(dim=-1)
        print(
            f"speaker {spk:3d}: {len(groups[spk]):3d} clips averaged  norm={emb.norm():.4f}  "
            f"per-clip spread mean={spread.mean():.4f} max={spread.max():.4f}"
        )

    table = torch.cat(rows)  # [N, 768]; a single speaker gives [1, 768], which the loader also accepts
    print(f"encoder input    : {raw_sr} Hz")
    print(f"embedding table  : {tuple(table.shape)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(table, args.out)
    print(f"written          : {args.out}")


if __name__ == "__main__":
    main()
