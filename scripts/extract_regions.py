#!/usr/bin/env python3
from .GTF import GTF
import sys

SEPARATOR = "\t"


def get_3prime_interval(transcript, length):
    if transcript.strand == "+" or transcript.strand == ".":
        start = transcript.end - (length // 2) + 1
        end = transcript.end + (length // 2)
    else:
        end = transcript.start + (length // 2) - 1
        start = transcript.start - (length // 2)
    return start, end


def get_5prime_interval(transcript, length):
    if transcript.strand == "+" or transcript.strand == ".":
        end = transcript.start + (length // 2)
        start = transcript.start - (length // 2) + 1
    else:
        start = transcript.end - (length // 2)
        end = transcript.end + (length // 2) - 1
    return start, end


def get_interval(transcript, length, region):
    return (
        get_3prime_interval(transcript, length)
        if region == 3
        else get_5prime_interval(transcript, length)
    )


def get_intervals(transcripts, length, region):
    intervals = []
    for transcript in transcripts:
        start, end = get_interval(transcript, length, region)
        if start > 0:
            intervals.append(
                (
                    transcript.seqname,
                    start - 1,
                    end,
                    transcript["transcript_id"],
                    transcript.score,
                    transcript.strand,
                )
            )
        else:
            print(
                f"{transcript['transcript_id']} skipped: Start Coordinate detected that is < 0.",
                file=sys.stderr,
            )
    return intervals


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-i",
        "--input",
        help="Path to your GTF file. Use stdin by default",
        type=argparse.FileType("r"),
        default=(None if sys.stdin.isatty() else sys.stdin),
    )
    parser.add_argument(
        "-l",
        "--length",
        help="",
        type=int,
        default=512,
    )
    parser.add_argument("-r", "--region", help="", type=int, choices=[5, 3], default=5)
    args = parser.parse_args()

    transcripts = []
    for gene in GTF.reconstruct_full_gtf(args.input):
        transcripts += gene.transcripts
    intervals = get_intervals(transcripts, args.length, args.region)

    for interval in intervals:
        print(SEPARATOR.join(map(str, interval)))
