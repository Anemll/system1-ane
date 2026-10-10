#!/usr/bin/env python3
"""Read-only inspection of cached Core AI placement, without loading ML runtimes.

Origin: Anemll/anemll-forge. Jeff placement audits use
``python coreai/inspect_coreai_cache.py --model-dir <coreai/> --executable python --strict``.
"""
import argparse
import hashlib
import json
import plistlib
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from qwen38_kv_cache import cache_format, cache_entries


def package_entries(model_dir, drafter=None, kv_cache_dtype='auto'):
    manifest = json.loads((model_dir / 'manifest.json').read_text())
    selected = cache_format(manifest, kv_cache_dtype)
    def target_record(record, entries=None):
        package = model_dir / record['file']
        compiled = model_dir / record['compiled'] if record.get('compiled') else package.with_suffix('.aimodelc')
        return package, entries if entries is not None else record.get('entries', ['h8']), compiled
    result = [target_record(c, list(cache_entries(manifest, c, selected).values())) for c in manifest['chunks']]
    head = manifest['head']
    result.append(target_record(head))
    if drafter:
        # Extract actual drafter entry names from its matching sidecar.
        sidecar = drafter.with_suffix('.json')
        meta = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        entries = meta.get('entries')
        if isinstance(entries, dict):
            entries = list(entries)
        if not isinstance(entries, list) or not all(isinstance(x, str) for x in entries):
            entries = []  # Unknown contract; never invent entries and pass the audit.
        result.append((drafter, entries, None))  # Drafter loads its explicit package directly.
    return result


def inspect_package(package, expected, cache_root, os_build, executable, compiled_override=None):
    hashfile = package / 'main.hash'
    result = {'package': str(package), 'expected_entries': expected,
              'status': 'unknown', 'specializations': []}
    if compiled_override is not None and compiled_override.exists():
        result.update(reason='compiled override exists; selected artifact is outside this source-cache audit',
                      compiled_override=str(compiled_override))
        return result
    if not hashfile.is_file():
        result['reason'] = 'missing source main.hash'
        return result
    data = hashfile.read_bytes()
    result.update(source_main_hash_hex=data.hex(), main_hash_sha256=hashlib.sha256(data).hexdigest())
    metadata = package / 'metadata.json'
    if metadata.is_file():
        result['metadata_sha256'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    cache = cache_root / os_build / Path(executable).name.replace('_', '-') / data.hex()
    result['cache_directory'] = str(cache)
    for mf in sorted(cache.glob('*/model.aimodelx/**/manifest.plist')):
        if '.mpsgraphpackage' not in str(mf):
            continue
        item = {'manifest': str(mf), 'entries': {}, 'graphs': [], 'status': 'unknown'}
        try:
            versions = plistlib.loads(mf.read_bytes()).get('Package Version', {})
            records = []
            missing_graph = False
            item['compile_modes'] = sorted(set(int(x) for x in re.findall(rb'aneBondedCompileMode\W+(\d+)', mf.read_bytes())))
            for version, fields in versions.items():
                for module in fields.get('Optimized Modules', {}).values():
                    attrs = module.get('Entry Function Attributes', {})
                    ane_symbols = set()
                    filename = module.get('File Name')
                    if filename:
                        graph = (mf.parent / filename).resolve()
                        if not graph.is_relative_to(mf.parent.resolve()):
                            raise ValueError('graph path escapes cache package')
                        if graph.is_file():
                            blob = graph.read_bytes()
                            ane_symbols = {x.decode('ascii') for x in re.findall(rb'[A-Za-z0-9_-]+_ANE_region_[A-Za-z0-9_]+', blob)}
                            item['graphs'].append({'file': str(graph), 'bytes': len(blob),
                                'sha256': hashlib.sha256(blob).hexdigest(),
                                'gpu_region_name_count': len(set(re.findall(rb'[A-Za-z0-9_-]+_GPU_region_[A-Za-z0-9_]+', blob))),
                                'ane_region_name_count': len(ane_symbols)})
                        else:
                            missing_graph = True
                    else:
                        missing_graph = True
                    records.extend((name, a, version, any(s.startswith(name + '_ANE_region_') for s in ane_symbols))
                                   for name, a in attrs.items())
            for entry in expected:
                found = [(n, a, v, recognized) for n, a, v, recognized in records if n == entry or n.startswith(entry + '_')]
                full = bool(found) and all(isinstance(a, list) and 'mps.fullyPlacedOnANE' in a
                                          and 'mps.noGPUActivity' in a and recognized
                                          for _, a, _, recognized in found)
                item['entries'][entry] = {'status': 'fully_ane' if full else 'unknown',
                    'matching_symbols': [n for n, _, _, _ in found]}
            gpu = any(g['gpu_region_name_count'] for g in item['graphs'])
            if gpu:
                item['status'] = 'gpu_regions_present'
            elif (not missing_graph and item['graphs'] and all(g['ane_region_name_count'] for g in item['graphs'])
                  and expected and all(e['status'] == 'fully_ane' for e in item['entries'].values())):
                item['status'] = 'fully_ane'
        except (OSError, ValueError, TypeError, AttributeError, plistlib.InvalidFileException) as exc:
            item['reason'] = str(exc)
        result['specializations'].append(item)
    states = [x['status'] for x in result['specializations']]
    if 'gpu_regions_present' in states:
        result['status'] = 'gpu_regions_present'
    elif states and all(s == 'fully_ane' for s in states):
        result['status'] = 'fully_ane'
    elif not states:
        result['reason'] = 'no matching cached MPSGraph manifest'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--drafter', type=Path)
    parser.add_argument('--kv-cache-dtype', choices=('auto', 'fp16', 'v8', 'kv8'), default='auto',
                        help='inspect selected functions in a shared-weight selectable export')
    parser.add_argument('--cache-root', type=Path, default=Path.home() / 'Library/Caches/coreai-cache')
    parser.add_argument('--os-build', help='defaults to sw_vers -buildVersion')
    parser.add_argument('--executable', default=sys.executable, help='cache executable identity; basename with underscores replaced by hyphens')
    parser.add_argument('--strict', action='store_true', help='exit 1 unless every inspected package is fully_ane; missing/unknown also fail')
    args = parser.parse_args(argv)
    try:
        build = args.os_build or subprocess.run(['sw_vers', '-buildVersion'], check=True, capture_output=True, text=True).stdout.strip()
        if not build or '/' in build or build in ('.', '..'):
            raise ValueError('invalid OS build')
        packages = package_entries(args.model_dir.expanduser(), args.drafter.expanduser() if args.drafter else None,
                                   args.kv_cache_dtype)
        results = [inspect_package(p, e, args.cache_root.expanduser(), build, args.executable, compiled)
                   for p, e, compiled in packages]
        report = {'os_build': build, 'executable_identity': Path(args.executable).name.replace('_', '-'),
                  'scope': 'cached specializations, not proof of live hardware execution or numerical correctness',
                  'packages': results}
        print(json.dumps(report, indent=2))
        return int(args.strict and any(x['status'] != 'fully_ane' for x in results))
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
