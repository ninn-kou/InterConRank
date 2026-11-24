"""Stream publisher downloads into the dataset's raw directory."""

from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen

from tqdm.auto import tqdm


def download_source(target, spec):
    """Handle whole-file responses and publishers that serve one range at a time."""
    temporary = NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.",
                                  suffix=".part", delete=False)
    part = Path(temporary.name)
    try:
        with temporary as stream, tqdm(total=spec["bytes"], desc=target.name,
                                       unit="B", unit_scale=True, unit_divisor=1000) as bar:
            while True:
                offset = stream.tell()
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(spec["url"], headers=headers), timeout=90) as response:
                    if response.status == 206:
                        span = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",
                                            response.headers.get("Content-Range", ""))
                        if not span or int(span[1]) != offset:
                            raise OSError(f"Cannot continue the download of {target.name}")
                        end = int(span[2]) + 1
                        more = end < int(span[3])
                    else:
                        # A server may send the entire file in response to a range request.
                        if offset:
                            stream.seek(0)
                            stream.truncate()
                            bar.reset()
                        end = response.length
                        more = False
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        stream.write(chunk)
                        bar.update(len(chunk))
                    if end is not None and stream.tell() != end:
                        raise OSError(f"Interrupted download of {target.name}")
                    if not more:
                        break
        part.replace(target)
    finally:
        part.unlink(missing_ok=True)
