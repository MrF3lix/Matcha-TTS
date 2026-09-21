"""Turn a multi-speaker corpus's metadata table into Matcha filelists.

Matcha's filelist format is one utterance per line,

    <audio path>|<text>                  single speaker  (n_spks: 1)
    <audio path>|<speaker id>|<text>     multi-speaker   (n_spks: N)

with speaker ids contiguous 0..N-1: they index `nn.Embedding(n_spks, ...)` in the acoustic model
and the [N, 768] VocBulwark embedding table. This script reads a CSV/TSV with an audio column, a
text column and (for multi-speaker) a speaker column, assigns ids, checks the data, and writes
train/val filelists plus, for multi-speaker, a `speakers.json` mapping id -> speaker name.

    # single speaker, LJSpeech-style `id|text|...` without header (the Bernese corpus)
    python scripts/build_filelist.py \
        --metadata data/be/metadata.txt --no-header --audio-col 0 --text-col 1 \
        --audio-root data/be/prepared/wav --out-dir data/be/filelists --check-audio

    # multi-speaker
    python scripts/build_filelist.py \
        --metadata /path/to/corpus/metadata.csv --audio-root /path/to/corpus/wavs \
        --audio-col file --speaker-col speaker --text-col text \
        --out-dir data/swissgerman/filelists

Audio paths are written exactly as `--audio-root` + file, so a relative root (e.g. `data/be/...`)
gives filelists that work on any machine with the same layout under the project root.

Checks performed (all reported, `--strict` turns them into an error):
  * every audio file exists (utterances with missing audio are dropped);
  * with --check-audio: sample rate, channel count and duration are read; clips longer than
    --max-seconds (the Whisper front-end truncates past MAX_AUDIO_SECONDS) or shorter than
    --min-seconds are dropped;
  * after running the cleaner, every character is in matcha/text/symbols.py. Anything else
    would crash text_to_sequence during training, so the offending characters and example
    utterances are listed; fix them in the metadata or extend the symbol set.

The split is per speaker (each speaker contributes to validation), so validation audio samples
and the speaker-embedding table cover everyone. The summary at the end prints the values to put
into configs/data/<name>.yaml (`n_spks`) and the command for the mel statistics.
"""

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from matcha.text import _clean_text  # noqa: E402  pylint: disable=wrong-import-position
from matcha.text.symbols import symbols  # noqa: E402  pylint: disable=wrong-import-position


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", type=Path, required=True, help="CSV/TSV with one row per utterance")
    p.add_argument("--delimiter", default=None, help="Column delimiter (default: sniffed; use '\\t' for TSV, '|' for LJSpeech style)")
    p.add_argument("--no-header", action="store_true", help="Metadata has no header row; columns are then 0-based indices")
    p.add_argument("--audio-col", required=True, help="Column with the audio path or file stem")
    p.add_argument("--speaker-col", default=None, help="Column with the speaker name/id (omit for a single-speaker corpus)")
    p.add_argument("--text-col", required=True, help="Column with the transcript")
    p.add_argument("--audio-root", type=Path, default=None, help="Prefix for relative audio paths")
    p.add_argument("--audio-ext", default=".wav", help="Appended when the audio column has no extension (default .wav)")
    p.add_argument("--cleaners", nargs="+", default=["swiss_german_cleaners"], help="Cleaners to validate against")
    p.add_argument("--out-dir", type=Path, required=True, help="Where train.txt, val.txt, speakers.json go")
    p.add_argument("--val-fraction", type=float, default=0.02, help="Fraction of each speaker's utterances for validation")
    p.add_argument("--min-val-per-speaker", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--check-audio", action="store_true", help="Open every file to read rate/duration (slower)")
    p.add_argument("--min-seconds", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--strict", action="store_true", help="Exit non-zero if anything was dropped or any character is unknown")
    return p.parse_args()


def read_rows(args):
    with open(args.metadata, encoding="utf-8", newline="") as f:
        sample = f.read(64 * 1024)
        f.seek(0)
        delimiter = args.delimiter
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",\t|;").delimiter
            except csv.Error:
                delimiter = ","
        delimiter = delimiter.replace("\\t", "\t")
        # Transcripts contain quotation marks; never let the csv module treat them as quoting.
        if args.no_header:
            reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE)
            rows = [{str(i): v for i, v in enumerate(r)} for r in reader if r]
        else:
            rows = list(csv.DictReader(f, delimiter=delimiter, quoting=csv.QUOTE_NONE))
    if not rows:
        raise SystemExit(f"{args.metadata}: no rows")
    for col in (args.audio_col, args.speaker_col, args.text_col):
        if col is not None and col not in rows[0]:
            raise SystemExit(f"column {col!r} not found; available: {list(rows[0])}")
    print(f"read {len(rows)} rows from {args.metadata} (delimiter {delimiter!r})")
    return rows


def resolve_audio(value, args):
    path = Path(value.strip())
    if not path.suffix:
        path = path.with_suffix(args.audio_ext)
    if not path.is_absolute() and args.audio_root is not None:
        path = args.audio_root / path
    return path


def audio_info(path):
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    info = sf.info(str(path))
    return info.samplerate, info.channels, info.frames / info.samplerate


def main():
    args = parse_args()
    random.seed(args.seed)
    rows = read_rows(args)
    known = set(symbols)

    dropped = Counter()
    unknown_chars = defaultdict(list)  # char -> example texts
    rates, durations = Counter(), []
    per_speaker = defaultdict(list)  # speaker name -> [(path, cleaned text)]

    for row in rows:
        text = row[args.text_col].strip()
        speaker = row[args.speaker_col].strip() if args.speaker_col is not None else "speaker"
        path = resolve_audio(row[args.audio_col], args)
        if not text or not speaker:
            dropped["empty text or speaker"] += 1
            continue
        if not path.is_file():
            dropped["audio missing"] += 1
            continue
        if args.check_audio:
            sr, channels, dur = audio_info(path)
            rates[(sr, channels)] += 1
            if dur < args.min_seconds:
                dropped[f"shorter than {args.min_seconds}s"] += 1
                continue
            if dur > args.max_seconds:
                dropped[f"longer than {args.max_seconds}s"] += 1
                continue
            durations.append(dur)

        cleaned = _clean_text(text, args.cleaners)
        bad = {ch for ch in cleaned if ch not in known}
        for ch in bad:
            if len(unknown_chars[ch]) < 3:
                unknown_chars[ch].append(cleaned)
        if bad:
            dropped["unknown characters"] += 1
            continue
        # The filelist is split on "|", so the transcript must not contain it.
        per_speaker[speaker].append((str(path), text.replace("|", " ")))

    if not per_speaker:
        raise SystemExit("nothing survived the checks")

    speakers = sorted(per_speaker)
    multi = args.speaker_col is not None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for spk_id, name in enumerate(speakers):
        utts = per_speaker[name]
        random.shuffle(utts)
        n_val = max(args.min_val_per_speaker, round(len(utts) * args.val_fraction)) if len(utts) > 1 else 0
        for i, (path, text) in enumerate(utts):
            line = f"{path}|{spk_id}|{text}" if multi else f"{path}|{text}"
            (val if i < n_val else train).append(line)
    random.shuffle(train)
    random.shuffle(val)

    (args.out_dir / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.out_dir / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    if multi:
        (args.out_dir / "speakers.json").write_text(
            json.dumps({i: n for i, n in enumerate(speakers)}, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print()
    print(f"speakers ({len(speakers)}):")
    for spk_id, name in enumerate(speakers):
        print(f"  {spk_id:3d}  {name:<24s} {len(per_speaker[name]):6d} utterances")
    print(f"train: {len(train)}   val: {len(val)}   -> {args.out_dir}")
    if rates:
        print("audio (rate, channels):", dict(rates))
        total_h = sum(durations) / 3600
        print(f"duration: {total_h:.2f} h total, {min(durations):.2f}-{max(durations):.2f} s per clip")
    if dropped:
        print("dropped:", dict(dropped))
    if unknown_chars:
        print("\ncharacters not in matcha/text/symbols.py after cleaning (utterances with them were dropped):")
        for ch, examples in sorted(unknown_chars.items()):
            print(f"  {ch!r} U+{ord(ch):04X}   e.g. {examples[0][:70]!r}")
        print("fix the transcripts, or add the character to _letters_german in symbols.py and bump n_vocab.")

    print("\nnext:")
    print(f"  set   n_spks: {len(speakers)}   in configs/data/<your dataset>.yaml, then")
    print("  uv run matcha-data-stats -i <your dataset>.yaml   and paste mel_mean/mel_std into that yaml,")
    if multi:
        print(f"  uv run python scripts/compute_speaker_embedding.py --filelist {args.out_dir / 'train.txt'} --out <table>.pt")
    else:
        print("  uv run python scripts/compute_speaker_embedding.py --wav-dir <wav folder> --out <embedding>.pt")

    if args.strict and (dropped or unknown_chars):
        sys.exit(1)


if __name__ == "__main__":
    main()
