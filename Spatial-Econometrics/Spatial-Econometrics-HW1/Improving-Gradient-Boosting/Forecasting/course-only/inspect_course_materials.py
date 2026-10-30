"""Index only the supplied course slides; no external theory is introduced."""
import sys as _path_sys
from pathlib import Path as _Path
_path_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from notebook_api import REPOSITORY_ROOT, artifact_relative, resource_path

from pathlib import Path
import hashlib,json,re
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCES=[
 'Source/Lectures/Lecture-1/Слайды лекции1, часть1.pdf',
 'Source/Lectures/Lecture-1/Lecture-1.2.pdf',
 'Source/Lectures/Lecture-2/Lecture-2.1.pdf',
 'Source/Lectures/Lecture-2/Lecture-2.2.pdf',
 'Source/Lectures/Lecture-3/Lecture-3-2026-27.pdf',
 'Source/Lectures/Lecture-4/Lecture-4-2026-27.pdf',
 'Source/Lectures/Lecture-4/Lecture-4-part2-2026-27.pdf',
 'Source/Lectures/Lecture-5/Lecture9_4modul_2023-24.pdf']
PATTERNS={'SLX':r'\bSLX\b','SAR':r'\bSAR\b','SEM':r'\bSEM\b','SDM':r'\bSDM\b',
 'SDEM':r'\bSDEM\b','SARAR':r'\bSARAR\b','GWR':r'\bGWR\b|географически взвеш',
 'LISA':r'\bLISA\b','Moran':r'Moran|Моран','ridge':r'\bridge\b|ридж',
 'eigenvector_filtering':r'eigenvector spatial|spatial filtering|пространственн.{0,20}фильтр',
 'weights':r'матриц.{0,30}(вес|пространствен)|spatial weights'}

def main():
    rows=[];folder=HERE/'source-text';folder.mkdir(exist_ok=True)
    for relative in SOURCES:
        path=resource_path(relative);reader=PdfReader(path);parts=[];hits={key:[] for key in PATTERNS}
        for number,page in enumerate(reader.pages,1):
            text=page.extract_text() or '';parts.append('PAGE '+str(number)+'\n'+text)
            for key,pattern in PATTERNS.items():
                if re.search(pattern,text,re.I):hits[key].append(number)
        target=folder/(path.parent.name+'_'+path.stem+'.txt');target.write_text('\n\n'.join(parts),encoding='utf-8')
        rows.append(dict(source=relative,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pages=len(reader.pages),
            method_page_hits=hits,text_file=target.relative_to(HERE).as_posix()))
    (HERE/'lecture_index.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['source','pages','method_page_hits']} for r in rows],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
