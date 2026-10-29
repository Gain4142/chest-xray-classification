"""Resume selected NIH PNG downloads using byte ranges of the Kaggle ZIP.
No full archive is saved. Existing files are verified before reuse.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import struct
import time
import zipfile
import zlib
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'data/raw/images'
REPORT = ROOT/'outputs/metrics/dataset_qc'
API = 'https://www.kaggle.com/api/v1/datasets/download/nih-chest-xrays/data'


def request_range(url, start, end):
    for attempt in range(5):
        try:
            response = requests.get(url, headers={'Range':f'bytes={start}-{end}', 'Accept-Encoding':'identity'}, timeout=(20,120))
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end-start+1:
                raise ValueError('Server did not return the requested byte range.')
            return response.content
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
    def seekable(self):
        return True
    def readable(self):
        return True
    def tell(self):
        return self.position
    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position
    def read(self, n=-1):
        if n < 0:
            n = self.size-self.position
        n = min(n,self.size-self.position)
        if n <= 0:
            return b''
        data = request_range(self.url,self.position,self.position+n-1)
        self.position += len(data)
        return data


def inspect(path, expected_crc=None):
    data = path.read_bytes()
    if expected_crc is not None and zlib.crc32(data)&0xffffffff != expected_crc:
        raise ValueError('CRC mismatch')
    with Image.open(io.BytesIO(data)) as im:
        im.verify()
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        return {'image':path.name, 'bytes':len(data), 'width':im.width, 'height':im.height, 'format':im.format, 'mode':im.mode, 'sha256':hashlib.sha256(data).hexdigest(), 'decode_ok':True}


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    manifests = [pd_read(name) for name in ['train','val','test']]
    selected = {image:split for split,images in manifests for image in images}
    if sum(len(images) for _,images in manifests) != len(selected):
        raise ValueError('Image names overlap across manifests.')
    print(f'Preparing ZIP index for {len(selected):,} selected images...',flush=True)
    with requests.get(API,headers={'Range':'bytes=-64'},stream=True,timeout=60) as r:
        r.raise_for_status()
        if r.status_code != 206:
            raise ValueError('ZIP range access unavailable.')
        url = r.url
        size = int(r.headers['Content-Range'].split('/')[-1])
    with zipfile.ZipFile(RemoteZip(url,size)) as z:
        entries = sorted(z.infolist(),key=lambda x:x.header_offset)
        chosen = [x for x in entries if Path(x.filename).name in selected and not x.is_dir()]
        if len(chosen) != len(selected) or len({Path(x.filename).name for x in chosen}) != len(selected):
            raise ValueError('ZIP filename coverage/uniqueness mismatch.')
        next_offset = {x.header_offset:entries[i+1].header_offset if i+1<len(entries) else z.start_dir for i,x in enumerate(entries)}
    total_bytes = sum(x.file_size for x in chosen)
    print(f'Index verified. Original selected PNG size: {total_bytes/1e9:.2f} GB.',flush=True)
    missing=[]
    rows=[]
    for entry in chosen:
        target=DEST/Path(entry.filename).name
        if target.exists():
            try:
                row=inspect(target,entry.CRC)
                row['split']=selected[target.name]
                rows.append(row)
                continue
            except Exception:
                pass
        missing.append(entry)
    groups=[]
    for entry in missing:
        start=entry.header_offset
        end=next_offset[start]-1
        if groups and start-groups[-1]['end'] <= 262144 and end-groups[-1]['start'] < 8*1024*1024:
            groups[-1]['end']=end
            groups[-1]['entries'].append(entry)
        else:
            groups.append({'start':start,'end':end,'entries':[entry]})
    def process(group):
        blob=request_range(url,group['start'],group['end'])
        results=[]
        for entry in group['entries']:
            offset=entry.header_offset-group['start']
            header=struct.unpack_from('<IHHHHHIIIHH',blob,offset)
            if header[0] != 0x04034b50:
                raise ValueError('Invalid ZIP local header')
            begin=offset+30+header[-2]+header[-1]
            payload=blob[begin:begin+entry.compress_size]
            if entry.compress_type == zipfile.ZIP_DEFLATED:
                data=zlib.decompress(payload,-15)
            elif entry.compress_type == zipfile.ZIP_STORED:
                data=payload
            else:
                raise ValueError('Unsupported ZIP compression')
            if len(data)!=entry.file_size or zlib.crc32(data)&0xffffffff != entry.CRC:
                raise ValueError('Decompressed size/CRC mismatch')
            target=DEST/Path(entry.filename).name
            temporary=target.with_suffix('.png.part')
            temporary.write_bytes(data)
            row=inspect(temporary,entry.CRC)
            temporary.replace(target)
            row['image']=target.name
            row['split']=selected[target.name]
            results.append(row)
        return results
    start_time=time.monotonic()
    errors=[]
    last_log=0
    def progress(state):
        report={'state':state,'expected_images':len(selected),'completed_images':len(rows),'failed_groups':len(errors),'selected_original_bytes':total_bytes,'elapsed_seconds':round(time.monotonic()-start_time,1)}
        (REPORT/'download_progress.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    progress('running')
    with ThreadPoolExecutor(max_workers=24) as pool:
        futures={pool.submit(process,group):group for group in groups}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                errors.append({'images':[Path(x.filename).name for x in futures[future]['entries']],'error':str(exc)})
            now=time.monotonic()
            if now-last_log>=20:
                progress('running')
                print(f'{len(rows):,}/{len(selected):,} images verified; failed groups: {len(errors)}; elapsed {now-start_time:.0f}s',flush=True)
                last_log=now
    import pandas as pd
    pd.DataFrame(rows).sort_values('image').to_csv(REPORT/'downloaded_image_qc.csv',index=False)
    (REPORT/'download_errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    progress('complete' if len(rows)==len(selected) and not errors else 'incomplete_rerun_to_resume')
    print(f'Finished: {len(rows):,}/{len(selected):,}. Failed groups: {len(errors)}.',flush=True)
    if errors:
        raise RuntimeError('Some ranges failed; rerun to resume verified files.')


def pd_read(name):
    import pandas as pd
    df=pd.read_csv(ROOT/f'data/processed/reduced_train_full_eval/{name}.csv')
    return name,df['Image Index'].tolist()


if __name__ == '__main__':
    main()
