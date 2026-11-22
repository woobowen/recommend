"""Audit the actual downloadable ZIP's CRC, safe paths and local Markdown links."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import stat
from urllib.parse import unquote, urlsplit
import zipfile

LINK = re.compile(r'(?<!!)\[[^\]\n]*\]\(<?([^ \t\)<>]+)>?(?:[ \t]+"[^"]*")?\)')
SECRET = re.compile(
    rb"(?im)(?:-----BEGIN (?:OPENSSH|RSA|EC|DSA|PGP) PRIVATE KEY-----|"
    rb"\bgh[pousr]_[a-z0-9]{30,}|\bgithub_pat_[a-z0-9_]{30,}|\bAKIA[A-Z0-9]{16})"
)


def verify(zip_path, markdown):
    digest = {}
    anchors = {}
    relative_links = []
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        assert infos and archive.testzip() is None
        members = {info.filename.rstrip("/") for info in infos}
        assert len(infos) == len(members) == len({n.casefold() for n in members})
        for info in infos:
            name = info.filename.rstrip("/")
            p = PurePosixPath(name)
            assert not p.is_absolute() and "\\" not in name and ":" not in name
            assert all(x not in {"", ".", ".."} for x in name.split("/"))
            assert not stat.S_ISLNK(info.external_attr >> 16)
            if info.is_dir():
                continue
            SHA = hashlib.sha256()
            with archive.open(info) as reader:
                carry = b""
                for block in iter(lambda: reader.read(1048576), b""):
                    assert not SECRET.search(carry + block), f"credential pattern at {name}"
                    carry = block[-200:]
                    SHA.update(block)
            digest[name] = SHA.hexdigest()

        def headings(name):
            if name not in anchors:
                text = archive.read(name).decode("utf-8")
                values = {}
                for line in text.splitlines():
                    match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
                    if match:
                        slug = re.sub(r"[^\w\-\s]", "", match.group(1).lower()).replace(" ", "-")
                        count = values.get(slug, 0)
                        values[slug] = count + 1
                result = {slug if i == 0 else f"{slug}-{i}" for slug, n in values.items() for i in range(n)}
                result.update(re.findall(r"""(?:id|name)=["']([^"']+)["']""", text))
                anchors[name] = result
            return anchors[name]

        selected = markdown or [
            n for n in sorted(members) if n.endswith(".md") and not n.startswith(("code/", "frozen/", "sources/"))
        ]
        assert "README.md" in selected
        directory_members = lambda p: any(name.startswith(p + "/") for name in members)
        for doc in selected:
            text = archive.read(doc).decode("utf-8")
            for link in LINK.findall(text):
                parsed = urlsplit(link)
                if parsed.scheme or parsed.netloc:
                    continue
                target = unquote(parsed.path)
                assert not target.startswith("/"), (doc, link)
                resolved = posixpath.normpath(posixpath.join(posixpath.dirname(doc), target)) if target else doc
                assert resolved in {".", "/"} or resolved in members or directory_members(resolved), (doc, link)
                fragment = unquote(parsed.fragment)
                if fragment and resolved.endswith(".md"):
                    assert fragment in headings(resolved), (doc, link, resolved, headings(resolved))
                relative_links.append({"from": doc, "link": link, "resolved": resolved})
    with zip_path.open("rb") as stream:
        zip_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"status":"PASS", "file_bytes":zip_path.stat().st_size, "sha256":zip_sha,
            "members":len(members), "crc":"PASS", "credential_pattern_scan":"PASS",
            "safe_unique_paths":"PASS", "markdown_files":selected,
            "relative_links":relative_links, "member_sha256":digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path)
    parser.add_argument("--markdown", action="append")
    args = parser.parse_args()
    print(json.dumps(verify(args.zip, args.markdown), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
