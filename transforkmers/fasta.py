class Seq:
    def __init__(self, id, seq, commentary):
        self.id = id
        self.seq = seq
        self.commentary = commentary

    def __str__(self) -> str:
        return f"{self.id}: {self.commentary[:15]} | {self.seq[:30]}..."

    def __getitem__(self, item) -> str:
        return self.seq[item]

    def __len__(self) -> int:
        return len(self.seq)

    def write(self, n=70):
        yield f">{self.id} {self.commentary}" + "\n"
        for i in range(0, len(self.seq), n):
            yield self.seq[i : i + n] + "\n"


class Fasta:
    def __init__(self, fd):
        self.sequences = dict([(seq.id, seq) for seq in self._parse_fasta(fd)])
        self.filename = fd.name.split("/")[-1]

    def __str__(self) -> str:
        l = "\n".join([str(seq) for seq in self.sequences])
        return self.filename + "\n" + l

    def __getitem__(self, item) -> Seq:
        return self.sequences[item]

    def __iter__(self):
        for value in self.sequences.values():
            yield value

    def __len__(self) -> int:
        return len(self.sequences)

    @staticmethod
    def _parse_fasta(fd):
        from itertools import groupby

        faiter = (x[1] for x in groupby(fd, lambda line: line[0] == ">"))
        for header in faiter:
            # drop the ">", and Extract id, commentary from header
            splited = header.__next__()[1:].strip().split(" ")
            id, commentary = splited[0], " ".join(splited[1:])

            # join all sequence lines to one.
            seq = "".join(s.strip() for s in faiter.__next__())

            yield Seq(id, seq, commentary)
