"""Prepare downloaded archives in DomainBed layout; run from repository root."""
import json
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path('domainbed/data')
ARCHIVES = ROOT / '.archives'


def unpack_zip(name, destination):
    with zipfile.ZipFile(ARCHIVES / name) as archive:
        for member in archive.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(member.filename)
            if '__MACOSX' not in path.parts:
                archive.extract(member, destination)


def prepare_domainnet_domain(domain):
    marker = ROOT / f'.domainnet_{domain}.extracted'
    if not marker.exists():
        print('Extract:', domain, flush=True)
        unpack_zip(domain + '.zip', ROOT / 'domain_net')
        marker.touch()


def prepare(name):
    marker = ROOT / f'.{name}.prepared'
    if marker.exists():
        print('Already prepared:', name, flush=True)
        return
    if name == 'PACS':
        unpack_zip('PACS.zip', ROOT)
    elif name == 'VLCS':
        staging = ROOT / '.vlcs_staging'
        staging.mkdir(exist_ok=True)
        with tarfile.open(ARCHIVES / 'VLCS.tar.gz') as archive:
            archive.extractall(staging, filter='data')
        # JiGen's "full" contains train+crossval; add test exactly once.
        # Map PASCAL to VOC to preserve DomainBed's C,L,S,V ordering.
        for source in (staging / 'VLCS').iterdir():
            domain = {'CALTECH':'Caltech101','LABELME':'LabelMe',
                      'SUN':'SUN09','PASCAL':'VOC2007'}[source.name]
            for split in ['full', 'test']:
                for path in (source / split).glob('*/*'):
                    if path.is_file():
                        target = ROOT / 'VLCS' / domain / path.parent.name / (split + '_' + path.name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
        shutil.rmtree(staging)
    elif name == 'DomainNet':
        for domain in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']:
            prepare_domainnet_domain(domain)
        removed = 0
        for line in Path('domainbed/misc/domain_net_duplicates.txt').read_text().splitlines():
            path = ROOT / 'domain_net' / line.strip()
            if path.is_file():
                path.unlink()
                removed += 1
        print('Removed duplicate images:', removed, flush=True)
    elif name == 'TerraIncognita':
        data = defaultdict(list)
        with tarfile.open(ARCHIVES / 'eccv_18_annotations.tar.gz') as archive:
            for member in archive:
                if member.name.endswith('.json'):
                    annotation = json.load(archive.extractfile(member))
                    for key, values in annotation.items():
                        data[key].extend(values)
        categories = {row['id']: row['name'] for row in data['categories']}
        selected_categories = {'bird','bobcat','cat','coyote','dog','empty','opossum','rabbit','raccoon','squirrel'}
        labels = defaultdict(set)
        for row in data['annotations']:
            category = categories[row['category_id']]
            if category in selected_categories:
                labels[row['image_id']].add(category)
        selected = defaultdict(set)
        for row in data['images']:
            location = str(row['location'])
            if location in {'38','46','100','43'}:
                for category in labels[row['id']]:
                    selected[row['file_name']].add(Path('location_' + location) / category / row['file_name'])
        found = set()
        with tarfile.open(ARCHIVES / 'eccv_18_all_images_sm.tar.gz', 'r|gz') as archive:
            for member in archive:
                relative = member.name.removeprefix('./').removeprefix('eccv_18_all_images_sm/')
                if relative in selected and member.isfile():
                    contents = archive.extractfile(member).read()
                    for target in selected[relative]:
                        path = ROOT / 'terra_incognita' / target
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(contents)
                    found.add(relative)
        assert found == set(selected), f'Missing {len(set(selected)-found)} Terra images'
        print('Terra selected images:', len(found), flush=True)
    else:
        raise ValueError(name)
    marker.touch()
    print('Prepared:', name, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=['PACS','VLCS','TerraIncognita','DomainNet'])
    for name in parser.parse_args().datasets:
        prepare(name)
