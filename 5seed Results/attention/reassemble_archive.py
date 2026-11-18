"""Reassemble and verify the downloadable attention ZIP."""

from hashlib import sha256
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parent
NAME = "Beauty_Toys_attention_5seed.zip"
EXPECTED = "9cdcd9bd4a7193b7e891f402fbe82b3defc5b597d6203be12f502d428f639272"
BLOCK = 1024 * 1024


def main():
    parts = [ROOT / "archive-parts" / f"{NAME}.part{i:02d}" for i in range(1, 7)]
    missing = [part.name for part in parts if not part.is_file()]
    if missing:
        raise SystemExit("Missing archive parts: " + ", ".join(missing))
    output = ROOT / NAME
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    digest = sha256()
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        temporary = Path(temp) / NAME
        with temporary.open("wb") as dest:
            for part in parts:
                with part.open("rb") as src:
                    for chunk in iter(lambda: src.read(BLOCK), b""):
                        digest.update(chunk)
                        dest.write(chunk)
        if digest.hexdigest() != EXPECTED:
            raise SystemExit("Archive SHA-256 mismatch")
        temporary.rename(output)
    print(f"Verified archive: {output}")


if __name__ == "__main__":
    main()
