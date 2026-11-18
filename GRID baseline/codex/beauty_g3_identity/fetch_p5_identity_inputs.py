"""Fetch exact original P5 Beauty identity inputs using HTTP Range ZIP reads.
Requires standard-library Python only. Downloads about 1.4 MB instead of 1.2 GB.
"""
import argparse
import hashlib
import io
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

class RemoteFile(io.RawIOBase):
    def __init__(self, file_id):
        self.url = ('https://drive.usercontent.google.com/download?' +
                    urllib.parse.urlencode(dict(id=file_id, export='download', confirm='t')))
        self.pos = 0
        self.requests = 0
        self.transferred = 0
        response = urllib.request.urlopen(urllib.request.Request(
            self.url, headers={'Range': 'bytes=-100000'}), timeout=45)
        assert response.status == 206
        self.length = int(response.headers['Content-Range'].split('/')[-1])
        self.tail = response.read()
        self.transferred += len(self.tail)
        self.requests += 1

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.length)
        return self.pos

    def read(self, count=-1):
        if count < 0:
            count = self.length - self.pos
        if not count:
            return b''
        end = min(self.pos + count, self.length)
        if self.pos >= self.length - len(self.tail):
            result = self.tail[self.pos - (self.length-len(self.tail)):end-(self.length-len(self.tail))]
        else:
            request = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end-1}'})
            with urllib.request.urlopen(request, timeout=60) as response:
                assert response.status == 206, response.status
                result = response.read()
                assert len(result) == end-self.pos
                self.requests += 1
                self.transferred += len(result)
        self.pos += len(result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    expected={
        'datamaps.json':'79bce2481b59b88362621dd07843661a308819f243a00c67540c69b348123d3a',
        'sequential_data.txt':'226cce9c3105299ca0db9615d7d3fb32b3175e90da43100ae352599f0f0107b8',
    }
    remote=RemoteFile('1qGxgmx7G_WB7JE4Cn_bEcZ_o_NAJLE3G')
    with zipfile.ZipFile(remote) as archive:
        for name,digest in expected.items():
            data=archive.read('data/beauty/'+name)
            assert hashlib.sha256(data).hexdigest()==digest,name
            output=args.output/name
            if output.exists() and output.read_bytes()!=data:
                raise ValueError('Refusing to overwrite a different input: '+str(output))
            output.write_bytes(data)
            print(name,len(data),digest)

if __name__=='__main__':
    main()
