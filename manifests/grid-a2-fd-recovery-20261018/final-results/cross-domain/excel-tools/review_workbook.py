#!/usr/bin/env python3
"""Independent, bounded OOXML audit. Never imports or calls the builder.

Reads the actual workbook and already accepted local source files; writes only
the requested JSON receipt. Uses Fraction arithmetic, not the builder's Decimal
calculation. No Git command, network, model, dataset or prediction execution.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path, PurePosixPath
import re
import unicodedata
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import zipfile

NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
RNS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PNS = {'p': 'http://schemas.openxmlformats.org/package/2006/relationships'}
FIXED = 'd5c0db4203d6226a7aa5a73583da65f2b00d125d'
PREFIX = 'manifests/grid-a2-fd-recovery-20261018/final-results/cross-domain/'
M = ['recall@5', 'ndcg@5', 'recall@10', 'ndcg@10']
COLS = ['K', 'N', 'Q', 'T']
COUNTS = {'Beauty': 22363, 'Toys': 19412, 'Sports': 35598}
ROUTES = {
    'Beauty': ['Beauty-A', 'Beauty-B'],
    'Toys': ['Toys-B', 'Toys-A-5300', 'Toys-A-5000'],
    'Sports': ['Sports-B', 'Sports-A1', 'Sports-A2'],
}
PAPER_ORIGINAL_ORDER = {
    'Beauty': {'RK-Means': '.0422 .0639 .0277 .0347', 'R-VQ': '.0422 .0638 .0282 .0351', 'RQ-VAE': '.0404 .0593 .0268 .0329'},
    'Toys': {'RK-Means': '.0376 .0577 .0243 .0308', 'R-VQ': '.0327 .0493 .0209 .0262', 'RQ-VAE': '.0342 .0514 .0224 .0280'},
    'Sports': {'RK-Means': '.0236 .0353 .0153 .0191', 'R-VQ': '.0234 .0352 .0151 .0189', 'RQ-VAE': '.0205 .0312 .0132 .0166'},
}
# Source identities already verified at fixed d5 by the preceding independent
# read-only table audit. They are checked again as local bytes, without Git.
INPUT_SHA = {
    PREFIX + 'cross-domain-original-metrics-with-a2.csv': '64574e2f2daa49375056059edf59309dffaaf769fc4d6102879c18be3583c43c',
    PREFIX + 'cross-domain-evidence-index-with-a2.json': 'b4688b3154d902eda167e535ec063d0d7997e2598d3b9322a5097f1ed9ce4f80',
    'manifests/beauty-extensions/runs/beauty-mlp-formal-v1/test/metrics.json': 'e70517004822de47058906de13fa53609b34740b1aa344162cd5121a5b00bbb2',
    'manifests/beauty-extensions/runs/beauty-mlp-formal-v1/training_result.json': '320147c08fec70379ffbb70985f59fa9c390bc558d9918fb27bfcb164b490653',
    'manifests/grid-formal-20261015/runs/gf-20261015-toys-mlp-formal-v1/output/RESULT.json': '0cfe300ff99f9b2c86aa9b1e33e58db2f86c3476534d7cc27cb74118af03da82',
    'manifests/grid-formal-20261015/runs/gf-20261015-sports-mlp-formal-v1/output/RESULT.json': 'a3b0c522e6a980f1c591de28a914b0ab68ae3f6d6597b9fb5132a2505e161a42',
    'manifests/grid-formal-20261015/FINAL_RECEIPT.md': '192c4e5d5e398a321a11dd7b6910a24dfcb4909ce6122d513b3654064d36f79d',
}
EVAL_COMMIT = {
    **{r: '2cf4c793c2e0b2e9dc7e7a9269f5e54d5c667896' for r in ['Beauty-A', 'Beauty-B']},
    **{r: 'c8acac42d5e13ca80af75faac5d1f7bc33fb589f' for r in ['Toys-B', 'Sports-B']},
    **{r: 'b44399926c98369191c2ff5c2425377c8c07d3c8' for r in ['Toys-A-5300', 'Toys-A-5000', 'Sports-A1']},
    'Sports-A2': '85b77f5e2a5b497fdc1f620649ba488fc783a245',
}
MLP_COMMIT = {'Beauty': '89b6d961c81333b0a44c4eb17527ced11897849f', 'Toys': '4495c38459971aee36b00d881eb50f50fafa3159', 'Sports': '4495c38459971aee36b00d881eb50f50fafa3159'}
STEPS = {'Beauty-A': 4800, 'Beauty-B': 12500, 'Toys-B': 6200, 'Toys-A-5300': 5300, 'Toys-A-5000': 5000, 'Sports-B': 9500, 'Sports-A1': 8600, 'Sports-A2': 6900}
MLP_TRAIN = {'Beauty': (32, 64, 6400, 2600, 3600, 36), 'Toys': (32, 48, 4800, 4400, 5400, 54), 'Sports': (16, 16, 1600, 4900, 5900, 59)}


def digest(b):
    return hashlib.sha256(b).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def index_col(col):
    n = 0
    for c in col:
        n = n * 26 + ord(c) - 64
    return n


def col_index(n):
    s = ''
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def coordinate(ref):
    m = re.fullmatch(r'([A-Z]+)([1-9][0-9]*)', ref)
    require(m is not None, 'Invalid cell address ' + ref)
    return index_col(m[1]), int(m[2])


def rectangle(ref):
    a, b = ref.split(':') if ':' in ref else (ref, ref)
    x1, y1 = coordinate(a)
    x2, y2 = coordinate(b)
    require(x1 <= x2 and y1 <= y2, 'Invalid rectangle ' + ref)
    return x1, y1, x2, y2


def overlap(a, b):
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def fraction_string(value):
    return str(value.numerator) if value.denominator == 1 else str(value.numerator) + '/' + str(value.denominator)


def url(path, ref=FIXED):
    return 'https://github.com/woobowen/recommend/blob/' + ref + '/' + path


def check(root, workbook, a1_raw):
    inputs = []

    def read(path, pinned=False):
        b = (root / path).read_bytes()
        if pinned:
            require(digest(b) == INPUT_SHA[path], 'Fixed source identity mismatch: ' + path)
        inputs.append({'path': path, 'bytes': len(b), 'sha256': digest(b), 'fixed_d5_sha_checked': pinned})
        return b

    def obj(path):
        return json.loads(read(path, path in INPUT_SHA), parse_float=str)

    csv_path = PREFIX + 'cross-domain-original-metrics-with-a2.csv'
    index_path = PREFIX + 'cross-domain-evidence-index-with-a2.json'
    source_rows = list(csv.DictReader(io.StringIO(read(csv_path, True).decode())))
    require(len(source_rows) == 24, 'Expected 24 complete route/condition records')
    source = {(r['route'], r['condition']): r for r in source_rows}
    require(len(source) == 24, 'Duplicate source route/condition')
    index = obj(index_path)
    routes = {r['route_id']: r for r in index['routes']}
    require(len(routes) == 8 and index['available_condition_count'] == 24 and index['pending_condition_count'] == 0, 'Incomplete fixed cross-domain index')
    mlp = {}
    beauty_path = 'manifests/beauty-extensions/runs/beauty-mlp-formal-v1/test/metrics.json'
    beauty = obj(beauty_path)
    require(beauty['formal'] and beauty['complete_test_coverage'] and beauty['bounded_test_batches'] is None and beauty['samples'] == 22363, 'Beauty MLP is not complete formal test')
    for d in COUNTS:
        path = 'manifests/beauty-extensions/runs/beauty-mlp-formal-v1/training_result.json' if d == 'Beauty' else 'manifests/grid-formal-20261015/runs/gf-20261015-' + d.lower() + '-mlp-formal-v1/output/RESULT.json'
        t = obj(path)
        require(t['status'] == 'COMPLETE' and not t['engineering_only'], d + ' MLP incomplete/engineering-only')
        require(all(t[k] for k in ['formal', 'training_complete', 'test_complete', 'fresh_initialization', 'full_parameter_training']), d + ' MLP flags')
        test = t['test_metrics']
        require(test['status'] == 'PASS' and test['coverage_exact'] and test['samples'] == COUNTS[d], d + ' MLP coverage')
        values = {m: F(beauty['metrics'][0]['test/' + m] if d == 'Beauty' else test['metrics'][m]) for m in M}
        raw, accum, interval, best, final, validations = MLP_TRAIN[d]
        require(t['global_step'] == final and t['completed_full_validations'] == validations, d + ' MLP training facts')
        require(('step=' + str(best).zfill(6)) in t['best_checkpoint']['path'], d + ' MLP best step')
        mlp[d] = {'train': t, 'metrics': values, 'source': beauty_path if d == 'Beauty' else path}
    receipt = read('manifests/grid-formal-20261015/FINAL_RECEIPT.md', True).decode()
    # Exact integer rounding here is independent of Decimal formatting.
    def rounded9(x):
        a = x * 10**9
        q, r = divmod(a.numerator, a.denominator)
        q += int(2*r > a.denominator or (2*r == a.denominator and q % 2))
        s = str(q).zfill(10)
        return s[:-9] + '.' + s[-9:]
    for d in COUNTS:
        line = next(s for s in receipt.splitlines() if s.startswith('| ' + d.lower() + ' | 0.'))
        require([rounded9(mlp[d]['metrics'][m]) for m in M] == [s.strip() for s in line.split('|')[2:6]], d + ' FINAL_RECEIPT rounding')

    paper = {d: {method: dict(zip(M, [F(vals.split()[i]) for i in [0, 2, 1, 3]])) for method, vals in methods.items()} for d, methods in PAPER_ORIGINAL_ORDER.items()}
    expected = {}
    for d in COUNTS:
        for method in ['RK-Means', 'R-VQ', 'RQ-VAE']:
            ident = d + '.Paper.' + method
            expected[ident] = {'result': ident, 'domain': d, 'route': method, 'condition': 'Paper', 'category': '论文', 'samples': '', 'ref': '', 'metrics': paper[d][method]}
        for route in ROUTES[d]:
            conditions = routes[route]['conditions']
            require(len(conditions) == 3 and {c['condition'] for c in conditions} == {'Normal', 'Shuffle3', 'Shuffle4'}, route + ' conditions')
            require(all(len({c[k]['sha256'] for c in conditions}) == 1 for k in ['sid', 'checkpoint']), route + ' cross-condition asset mismatch')
            for condition in ['Normal', 'Shuffle3', 'Shuffle4']:
                c = next(c for c in conditions if c['condition'] == condition)
                s = source[(route, condition)]
                require(s['result_status'] == c['result_status'] == 'COMPLETE', route + ' incomplete')
                require(int(s['samples']) == c['samples'] == c['unique_users'] == COUNTS[d], route + ' sample count')
                require(c['scientific_evaluation_execution']['commit'] == EVAL_COMMIT[route], route + ' execution commit')
                values = {m: F(s[m]) for m in M}
                require(all(values[m] == F(c['metrics_original'][m]) for m in M), route + ' index/CSV metric discrepancy')
                ident = d + '.' + route[len(d)+1:] + '.' + condition
                expected[ident] = {'result': ident, 'domain': d, 'route': route[len(d)+1:], 'condition': condition, 'category': 'Baseline' if condition == 'Normal' else 'Shuffle', 'samples': COUNTS[d], 'ref': d + '.Paper.RK-Means' if condition == 'Normal' else d + '.' + route[len(d)+1:] + '.Normal', 'metrics': values, 'evidence': c, 'csv': s, 'full_route': route}
        for reference in (ROUTES[d] if d != 'Sports' else ['Sports-B']):
            ident = d + '.MLP.vs.' + reference[len(d)+1:] + '.Normal'
            expected[ident] = {'result': d + '.MLP.Formal', 'domain': d, 'route': 'MLP', 'condition': 'MLP', 'category': 'MLP', 'samples': COUNTS[d], 'ref': d + '.' + reference[len(d)+1:] + '.Normal', 'metrics': mlp[d]['metrics'], 'mlp': mlp[d]}
    require(len(expected) == 39, 'Expected comparison records')
    b = workbook.read_bytes()
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) == 8, 'OOXML member count/duplicates')
        require(len({unicodedata.normalize('NFC', s).casefold() for s in names}) == len(names), 'OOXML case/Unicode path collision')
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name and '\\' not in name, 'Unsafe OOXML path')
        require(z.testzip() is None, 'ZIP CRC failed')
        parts = {name: ET.fromstring(z.read(name)) for name in names}
        members = [{'path': x.filename, 'bytes': x.file_size, 'crc32': f'{x.CRC:08x}', 'sha256': digest(z.read(x.filename))} for x in z.infolist()]
    wb = parts['xl/workbook.xml']
    sheets = wb.findall('s:sheets/s:sheet', NS)
    require(len(sheets) == 1 and sheets[0].get('name') == 'GRID实验总表' and sheets[0].get('state', 'visible') == 'visible', 'Exactly one visible worksheet')
    require(not any('externalLink' in s or 'vba' in s.casefold() for s in names), 'External workbook/macro part')
    calc = wb.find('s:calcPr', NS)
    require(calc is not None and calc.get('calcMode') == 'auto' and calc.get('fullCalcOnLoad') == '1' and calc.get('forceFullCalc') == '1', 'Automatic recalc flags')
    sheet = parts['xl/worksheets/sheet1.xml']
    table = parts['xl/tables/table1.xml']
    require(table.get('ref') == 'A8:AG47' and table.find('s:autoFilter', NS).get('ref') == 'A8:AG47', 'Table/filter bounds')
    require(len(sheet.findall('s:tableParts/s:tablePart', NS)) == 1, 'Exactly one actual Excel Table')
    pane = sheet.find('s:sheetViews/s:sheetView/s:pane', NS)
    require(pane is not None and all(pane.get(k) == v for k, v in {'xSplit': '6', 'ySplit': '8', 'topLeftCell': 'G9', 'state': 'frozen'}.items()), 'Freeze G9')
    cells, styles = {}, {}
    for row in sheet.findall('s:sheetData/s:row', NS):
        require(row.get('hidden', '0') != '1', 'Hidden row')
        for cell in row.findall('s:c', NS):
            address = cell.get('r')
            require(address not in cells, 'Duplicate cell address')
            coordinate(address)
            require(cell.get('t') != 'e', 'Cell cached error')
            inline = cell.find('s:is', NS)
            value = cell.find('s:v', NS)
            text = ''.join(inline.itertext()) if inline is not None else (value.text if value is not None else '')
            cells[address] = {'value': text, 'formula': cell.findtext('s:f', None, NS), 'type': cell.get('t'), 'style': int(cell.get('s', '0'))}
    def val(address):
        return cells.get(address, {}).get('value', '')
    xfs = parts['xl/styles.xml'].findall('s:cellXfs/s:xf', NS)
    nfs = {int(n.get('numFmtId')): n.get('formatCode') for n in parts['xl/styles.xml'].findall('s:numFmts/s:numFmt', NS)}
    def fmt(address):
        return nfs.get(int(xfs[cells[address]['style']].get('numFmtId')), '')
    headers = [val(col_index(i) + '8') for i in range(1, 34)]
    require(headers == [c.get('name') for c in table.findall('s:tableColumns/s:tableColumn', NS)] and len(set(headers)) == 33, 'Table/header column identities')
    require([headers[index_col(c)-1] for c in COLS] == ['Recall@5', 'NDCG@5', 'Recall@10', 'NDCG@10'], 'Requested metric column order')
    require(headers[0] == '比较记录ID' and headers[1] == '实验结果ID' and headers[7] == '对照记录ID', 'Stable ID columns')
    merges = [m.get('ref') for m in sheet.findall('s:mergeCells/s:mergeCell', NS)]
    rects = [rectangle(s) for s in merges]
    for i, rect in enumerate(rects):
        require(not overlap(rect, rectangle('A8:AG47')), 'Merge intersects table/header')
        require(not any(overlap(rect, other) for other in rects[:i]), 'Overlapping merged ranges')
    rels = {r.get('Id'): r for r in parts['xl/worksheets/_rels/sheet1.xml.rels']}
    links = {}
    for link in sheet.findall('s:hyperlinks/s:hyperlink', NS):
        address = link.get('ref')
        require(address not in links, 'Multiple hyperlinks in one cell')
        if link.get('location'):
            target = link.get('location')
            match = re.fullmatch(r"'GRID实验总表'!([A-Z]+[1-9][0-9]*)", target)
            require(match is not None and val(match[1]) != '', 'Invalid internal method target')
            links[address] = '#' + target
        else:
            rid = link.get('{' + RNS + '}id')
            require(rid in rels, 'Missing hyperlink relationship')
            rel = rels[rid]
            target = rel.get('Target')
            parsed = urlparse(target)
            require(rel.get('TargetMode') == 'External' and parsed.scheme == 'https' and parsed.hostname in ['github.com', 'arxiv.org'], 'Unsafe/non-HTTPS hyperlink')
            if parsed.hostname == 'github.com':
                require(re.fullmatch(r'/woobowen/recommend/(blob|commit)/[0-9a-f]{40}(/.+)?', parsed.path) is not None, 'GitHub link lacks fixed commit')
            links[address] = target
    record_rows = {val('A' + str(row)): row for row in range(9, 48)}
    require(len(record_rows) == 39 and set(record_rows) == set(expected), 'Comparison IDs/39 data rows')
    require(len({val('B' + str(row)) for row in range(9, 48)}) == 36, 'Distinct scientific results')
    checked_cells, max_rel_error, method_bindings = [], F(0), []
    formula_count = 0
    for ident, e in expected.items():
        row = record_rows[ident]
        for col, value in [('B', e['result']), ('C', e['domain']), ('D', e['category']), ('E', e['route']), ('F', e['condition']), ('G', str(e['samples'])), ('H', e['ref'])]:
            require(val(col + str(row)) == value, ident + ' identity/coverage/reference ' + col)
        for m, col in zip(M, COLS):
            address = col + str(row)
            require(cells[address]['type'] is None and cells[address]['formula'] is None and F(val(address)) == e['metrics'][m], ident + ' original numeric metric ' + m)
            require(fmt(address) == ('0.0000' if e['category'] == '论文' else '0.000000'), ident + ' metric display format')
            abs_addr = col_index(index_col(col)+1) + str(row)
            rel_addr = col_index(index_col(col)+2) + str(row)
            if not e['ref']:
                require(val(abs_addr) == val(rel_addr) == '' and cells[abs_addr]['formula'] is None and cells[rel_addr]['formula'] is None, 'Paper has a fabricated comparison')
                continue
            require(e['ref'] in record_rows, 'Missing reference ID')
            base = expected[e['ref']]['metrics'][m]
            delta = e['metrics'][m] - base
            ratio = delta / base
            lookup = 'INDEX($' + col + '$9:$' + col + '$47,MATCH($H' + str(row) + ',$A$9:$A$47,0))'
            require(cells[abs_addr]['formula'] == 'IFERROR(' + address + '-' + lookup + ',"")', ident + ' absolute formula ' + m)
            require(cells[rel_addr]['formula'] == 'IFERROR((' + address + '-' + lookup + ')/' + lookup + ',"")', ident + ' ratio formula ' + m)
            require(F(val(abs_addr)) == delta, ident + ' absolute cache ' + m)
            error = abs(F(val(rel_addr)) - ratio)
            max_rel_error = max(max_rel_error, error)
            require(error < F(1, 10**48), ident + ' ratio cache/units ' + m)
            require(fmt(abs_addr) == '[Blue]+0.00000000;[Red]-0.00000000;0.00000000', 'Absolute eight-place/sign format')
            require(fmt(rel_addr) == '[Blue]+0.00%;[Red]-0.00%;0.00%', 'Ratio percentage/sign format')
            checked_cells.append({'id': ident, 'metric': m, 'reference': e['ref'], 'original': val(address), 'abs_cache': val(abs_addr), 'ratio_cache': val(rel_addr), 'abs_formula_cell': abs_addr, 'ratio_formula_cell': rel_addr})
            formula_count += 2
        require('E' + str(row) in links and links['E' + str(row)].startswith('#'), ident + ' missing route navigation')
        if 'evidence' in e:
            c, s, route = e['evidence'], e['csv'], e['full_route']
            require(val('AD' + str(row)) == EVAL_COMMIT[route], ident + ' actual execution source')
            require(val('AC' + str(row)) == s['job_id'], ident + ' job ID')
            require(val('AA' + str(row)) == s['invalid_candidates'] and val('AB' + str(row)) == s['duplicate_candidates'], ident + ' candidate disclosures')
            require(c['checkpoint']['sha256'][:12] in val('Z' + str(row)) and str(STEPS[route]) in val('Z' + str(row)), ident + ' checkpoint label')
            require(c['sid']['sha256'][:12] in val('AF' + str(row)), ident + ' SID label')
            require(val('Y' + str(row)) == ('共享' if route.endswith('-B') else '双独立'), ident + ' SID table sharing')
            require(val('X' + str(row)) == ('R-Kmeans（配置支持）' if route.startswith('Toys-A') else 'R-Kmeans'), ident + ' quantization label')
            if route.startswith('Toys-A'):
                require(val('W' + str(row)) == '历史producer未唯一确定', ident + ' Toys producer limitation')
            source_link = (c.get('evaluation_result_source') or {}).get('github_url', url(index_path))
            require(links['AE' + str(row)] == source_link and links['AF' + str(row)] == url(index_path), ident + ' source hyperlinks')
        elif 'mlp' in e:
            d, t = e['domain'], e['mlp']['train']
            require(val('AD' + str(row)) == MLP_COMMIT[d] and val('Y' + str(row)) == '共享', ident + ' MLP source/sharing')
            require(val('X' + str(row)) == ('R-Kmeans' if d == 'Sports' else 'RQ-VAE+STE'), ident + ' MLP quantizer')
            require(t['best_checkpoint']['sha256'][:12] in val('Z' + str(row)), ident + ' MLP checkpoint')
            require((t.get('sid_asset') or t['sid'])['sha256'][:12] in val('AF' + str(row)), ident + ' MLP SID')
            require('非严格单因素' in val('J' + str(row)), ident + ' MLP reference boundary')
            require(links['AE' + str(row)] == url(e['mlp']['source']), ident + ' MLP raw source hyperlink')
    require(formula_count == 240 == sum(c['formula'] is not None for c in cells.values()), 'Exactly 120 abs plus 120 ratio formulas')
    require(len(checked_cells) == 120, '120 independent comparisons')
    require(Counter(val('C' + str(r)) for r in range(9, 48)) == {'Beauty': 11, 'Toys': 15, 'Sports': 13}, 'Domain row counts')
    mlp_result_ids = Counter(val('B' + str(r)) for r in range(9, 48) if val('D' + str(r)) == 'MLP')
    require(mlp_result_ids == {'Beauty.MLP.Formal': 2, 'Toys.MLP.Formal': 3, 'Sports.MLP.Formal': 1}, 'Six MLP references are only three models')
    # Read the notes themselves, not builder strings or its self-check results.
    notes_end = rectangle(sheet.find('s:dimension', NS).get('ref'))[3]
    require(notes_end >= 149 and notes_end <= 155, 'Bounded method appendix extent')
    note_rows = {r: (val('A' + str(r)), val('C' + str(r))) for r in range(51, notes_end + 1)}
    heads = {label.split('｜')[0]: r for r, (label, body) in note_rows.items() if '｜独立路线说明' in label or 'MLP｜一个正式模型' in label}
    require(len(heads) == 11, 'All eleven method sections')
    for route in [r for rr in ROUTES.values() for r in rr]:
        start = heads[route]
        text = '\n'.join(note_rows[r][1] for r in range(start, min(start + 6, notes_end + 1)))
        c = next(c for c in routes[route]['conditions'] if c['condition'] == 'Normal')
        for k in ['checkpoint', 'sid']:
            require(c[k]['path'] in text and c[k]['sha256'] in text, route + ' full method ' + k + ' binding')
        require(c['scientific_evaluation_execution']['commit'] in text and c['scientific_evaluation_execution']['snapshot_sha256'] in text, route + ' full method execution binding')
        if c.get('config'):
            require(c['config']['path'] in text and c['config']['sha256'] in text, route + ' source configuration identity')
        else:
            require('未复制配置资产字段' in text, route + ' absent config disclosure')
        require(str(COUNTS[route.split('-')[0]]) in text, route + ' method coverage')
        method_bindings.append({'route': route, 'section_row': start, 'checkpoint_sha256': c['checkpoint']['sha256'], 'sid_sha256': c['sid']['sha256'], 'evaluation_commit': c['scientific_evaluation_execution']['commit']})
    for d in COUNTS:
        start = heads[d + ' MLP']
        text = '\n'.join(note_rows[r][1] for r in range(start, start + 6))
        t = mlp[d]['train']
        for asset in [t['best_checkpoint'], t.get('sid_asset') or t['sid'], t.get('approved_config_asset') or t['config']]:
            require(asset['path'] in text and asset['sha256'] in text, d + ' MLP full asset identity')
        require(MLP_COMMIT[d] in text, d + ' MLP execution identity')
        raw, accum, interval, best, final, validations = MLP_TRAIN[d]
        for token in ['raw microbatch=' + str(raw), 'accumulate_grad_batches=' + str(accum), 'val_check_interval=' + str(interval), 'best step=' + str(best), 'final step=' + str(final), '完整验证' + str(validations) + '次', str(COUNTS[d]) + '个目标']:
            require(token in text, d + ' MLP training note ' + token)
        method_bindings.append({'route': d + '-MLP', 'section_row': start, 'checkpoint_sha256': t['best_checkpoint']['sha256'], 'sid_sha256': (t.get('sid_asset') or t['sid'])['sha256'], 'training_test_commit': MLP_COMMIT[d]})
    all_notes = '\n'.join(body for _, body in note_rows.values())
    for token in ['torch.rand(B,M,N).argsort(dim=2)', 'AuthorShuffleModel(_RuntimeBridge, OriginalShuffleModel)', 'super()', 'BaselineShuffleModel', 'encoder_forward_pass', 'Decoder lookup不受重排', '512→256→128', 'Dropout(0.1)', '第四码位mask', 'sep_token=None', '四位自回归', '5个seed平均', '非严格同配置复现', 'producer未唯一', 'best step8600', 'final step9600', 'best6900', 'final7900', '30 optimizer steps', FIXED]:
        require(token in all_notes, 'Required method/boundary statement: ' + token)
    # New A1 paragraph distinguishes the expanded effective config from its
    # base. Verify the narrow existing local originals, not the label alone.
    a1_files = {
        'used_config.json': ('1e3b608a2037bce5c2934a6cd220fd2ffc1c5b494fa617b4314fc672f4c6320a', 14202),
        'training_protocol.json': ('68ac81965cb3f13c7747ee50da912dfe8680438a0902fd5dd600ea70e2c0f661', 3501),
        'RESULT.json': ('9fe44ad7549ec4ae07f582f5799bbb728ece9c318a86c116cafcd85a7a0ada2e', 4387),
    }
    a1 = {}
    for name, (sha, size) in a1_files.items():
        path = a1_raw / name
        payload = path.read_bytes()
        require(digest(payload) == sha and len(payload) == size, 'A1 original identity: ' + name)
        a1[name] = json.loads(payload, parse_float=str)
        inputs.append({'path': str(path), 'bytes': len(payload), 'sha256': sha, 'scope': 'Existing accepted A1 originals; local narrow addendum only'})
    protocol_sha = '9d3823c2ada022f479fb3b8ac4e36bc2cf09267281db5b15edc636bec0a45c7e'
    protocol_payload = read('experiments/launch_20261017/training_protocol.json')
    require(digest(protocol_payload) == protocol_sha, 'Source A1/A2 training protocol hash')
    protocol = json.loads(protocol_payload, parse_float=str)
    require(a1['training_protocol.json']['protocol'] == protocol and a1['training_protocol.json']['asset']['sha256'] == protocol_sha, 'A1 wrapper versus actual protocol asset')
    cfg, result = a1['used_config.json'], a1['RESULT.json']
    require(protocol['seed'] == 42 and cfg['trainer']['precision'] == '32-true', 'A1 seed/precision')
    require(protocol['raw_microbatch'] == cfg['datamodule']['train_dataloader_config']['batch_size_per_device'] == 16, 'A1 raw batch')
    require(cfg['trainer']['accumulate_grad_batches'] == 16 and cfg['trainer']['val_check_interval'] == 1600, 'A1 accumulation/validation')
    require(cfg['model']['optimizer']['lr'] == '0.001' and cfg['model']['optimizer']['weight_decay'] == '0.0001' and cfg['model']['scheduler'] is None, 'A1 optimizer protocol')
    require(result['completed_full_validations'] == 96 and result['global_step'] == 9600 and result['full_validation_identity_exact'], 'A1 complete validation/final step')
    require(result['best_checkpoint']['sha256'] == expected['Sports.A1.Normal']['evidence']['checkpoint']['sha256'], 'A1 training-best identity')
    for token in ['14202字节', '1e3b608a2037bce5c2934a6cd220fd2ffc1c5b494fa617b4314fc672f4c6320a', protocol_sha, '96次完整验证', '实际展开used_config.json是另一文件']:
        require(token in all_notes, 'A1 effective/base distinction addendum: ' + token)
    require('原生表格软件' in all_notes and '实际结果另见excel-tools/excel-validation.json' in all_notes and '交付前另用现有LibreOffice' not in all_notes, 'No pre-signed application check')
    require('raw records' in all_notes and '不等于实际expanded-window数' in all_notes, 'Raw-record count versus expanded-window boundary')
    # Bounded source inspection establishes that the descriptions refer to the
    # existing implementations. This deliberately does not execute those files.
    source_tokens = {
        'experiments/beauty_extensions/author/models.py': ['class AuthorShuffleModel(_RuntimeBridge, OriginalShuffleModel)', 'super()._shuffle_hierarchical_embeddings', 'generator=generator', 'class AuthorMLPModel(_RuntimeBridge, OriginalMLPModel)'],
        'experiments/beauty_extensions/baseline.py': ['class BaselineShuffleModel(SemanticIDEncoderDecoder)', 'register_forward_hook(self._lookup)', 'NATIVE_SHUFFLE(self, output, self._encoder_mask, h, n)', 'self._encoder_mask = None'],
        'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_shuffle.py': ['torch.rand(batch_size, num_items, hierarchies_to_shuffle', 'torch.argsort(rand_values, dim=2)', 'if self.is_testing and self.shuffle_embedding_on_test:'],
        'references/wmh-original/GRID/src/models/modules/semantic_id/tiger_generation_model_mlp.py': ['self.sep_token = None', 'torch.nn.Linear(self.num_hierarchies * self.embedding_dim, self.embedding_dim * 2)', 'torch.nn.Dropout(0.1)', 'attention_mask = attention_mask[:, :, -1]', 'inputs_embeds_for_encoder = self.item_aggregation_mlp(flattened_embeds)'],
    }
    for path, tokens in source_tokens.items():
        text = read(path).decode()
        require(all(t in text for t in tokens), 'Method/source mismatch: ' + path)
    # Source-hash paragraph is also bound, rather than trusted as free text.
    for path, sha in INPUT_SHA.items():
        require(path + '\nSHA-256: ' + sha in all_notes, 'Source identity paragraph mismatch ' + path)
    sports_normal = expected['Sports.B.Normal']['evidence']
    sports_mlp = mlp['Sports']['train']
    require(sports_normal['sid']['sha256'] == sports_mlp['sid']['sha256'] and sports_normal['config']['sha256'] == sports_mlp['config']['sha256'], 'Sports primary reference SID/config mapping')
    require(sports_normal['checkpoint']['sha256'] != sports_mlp['best_checkpoint']['sha256'], 'Independent Sports weights disclosure')
    # Formula meaning is invariant under permutation because lookup reads unique
    # IDs. Test two independent reordered maps; this is not an Office UI sort.
    sorted_invariance = []
    for order in [list(reversed(list(record_rows))), sorted(record_rows, key=lambda k: (expected[k]['route'], expected[k]['domain'], k))]:
        remap = {ident: expected[ident]['metrics'] for ident in order}
        for ident, e in expected.items():
            if e['ref']:
                for m in M:
                    require((remap[ident][m] - remap[e['ref']][m]) == (e['metrics'][m] - expected[e['ref']]['metrics'][m]), 'Stable-ID sort arithmetic')
        sorted_invariance.append('PASS')
    builder_path = root / PREFIX / 'excel-tools/build_workbook.py'
    return {
        'schema': 'grid.excel.nonauthor.review.v1',
        'status': 'EXCEL_OOXML_NUMERIC_METHOD_NONAUTHOR_REVIEW_PASS',
        'reviewer': '/root/fd_repair_reviewer',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'authorship': 'Reviewer did not author or run build_workbook.py and did not modify the workbook.',
        'workbook': {'path': str(workbook.resolve()), 'bytes': len(b), 'sha256': digest(b)},
        'builder_read_only_identity': {'path': str(builder_path), 'bytes': builder_path.stat().st_size, 'sha256': digest(builder_path.read_bytes())},
        'review_implementation': {'path': str(Path(__file__).resolve()), 'sha256': digest(Path(__file__).read_bytes())},
        'fixed_scientific_source_commit': FIXED,
        'scope': 'Actual local OOXML ZIP, seven pinned d5 inputs, inherited accepted scientific identities and bounded static implementation inspection. No builder import/run, Git command, network, server, model, GPU, dataset or prediction evaluation.',
        'counts': {'sheets': 1, 'tables': 1, 'table_records': 39, 'distinct_results': 36, 'paper_records': 9, 'normal_records': 8, 'shuffle3_records': 8, 'shuffle4_records': 8, 'mlp_reference_records': 6, 'mlp_models': 3, 'original_numeric_metrics': 156, 'absolute_formulas_and_caches': 120, 'relative_formulas_and_caches': 120, 'independent_method_sections': 11, 'hyperlinks': len(links), 'merged_ranges': len(merges)},
        'layout': {'table': 'A8:AG47', 'freeze': 'G9', 'notes': 'A51:P' + str(notes_end), 'headers_match_table': True, 'merged_ranges_nonoverlapping_and_outside_table': True, 'link_targets_safe_and_internal_targets_exist': True},
        'numeric_checks': {'all_original_metrics_exact': True, 'all_absolute_caches_exact': True, 'relative_cache_max_absolute_error_fraction': fraction_string(max_rel_error), 'relative_cache_error_bound': '1e-48', 'relative_cache_unit': 'ratio, not percentage points; percent number format', 'source_metric_order': 'R5/N5/R10/N10', 'paper_column_reorder': 'R5/R10/N5/N10 to R5/N5/R10/N10', 'display': {'paper_places': 4, 'metric_places': 6, 'absolute_places': 8, 'relative_percent_places': 2}, 'FINAL_RECEIPT_9place_checks': 12, 'arithmetic': 'Fraction exact arithmetic, independent of builder Decimal'},
        'comparison_policy': {'normal': 'same-domain paper RK-Means; reference only', 'shuffle': 'own route Normal, same checkpoint/SID', 'MLP': 'Beauty A/B, Toys B/A5300/A5000, Sports B; six references but three result IDs; non-strict', 'stable_comparison_record_id_lookup': True, 'two_reordered_ID_map_checks': sorted_invariance},
        'test_target_counts': COUNTS,
        'zip_integrity': {'crc': 'PASS', 'all_xml_parsed': True, 'member_paths_unique_canonical_casefold': True, 'member_manifest': members},
        'scientific_identity_bindings': method_bindings,
        'sports_a1_effective_config_addendum': {'status': 'PASS', 'used_config_sha256': a1_files['used_config.json'][0], 'used_config_bytes': 14202, 'base_config_sha256': protocol['source_config_sha256'], 'protocol_asset_sha256': protocol_sha, 'protocol_wrapper_sha256': a1_files['training_protocol.json'][0], 'complete_validations': 96, 'base_and_used_config_are_distinct': True},
        'checked_comparisons': checked_cells,
        'inputs': inputs,
        'limitations': ['This is a delivery/table audit; prior scientific result acceptance is inherited, not repeated.', 'No Office GUI rendering or actual Office sort/recalculation was executed by this independent script; primary native-application readback is a separate receipt.', 'Hyperlink destinations are checked as fixed-commit URLs and local note targets; no fresh HTTP readback is claimed.', 'Publication and final application rendering/recalculation are not preapproved by this receipt.'],
        'new_installations': [],
        'workbook_modified': False,
        'scientific_files_modified': False,
        'publication_approved_by_this_receipt': False,
        'earlier_candidate_review': {'workbook_sha256': '24001c80ddb523ccfd15de17c5744563abdb00ba5a9ab73d65cbaea31e7e9a9b', 'receipt_sha256': '47651cb01fff222d7baac7c44976827ccae350d120a7cfaed6b2c6de52423b05', 'scope': 'Earlier candidate only; superseded by the actually audited workbook above.'},
        'review_tooling_note': 'A new literal guard initially expected the spelling expanded windows. The actual workbook accurately says expanded-window. The guard now checks the actual complete boundary phrase; no workbook or scientific value was changed by the reviewer.',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    base = here.parent.parent
    root = next(p for p in here.parents if (p / 'manifests').is_dir())
    ap.add_argument('--workbook', type=Path, default=base / 'GRID_Shuffle_MLP_Results_AllInOne_20261018.xlsx')
    ap.add_argument('--report', type=Path, default=base / 'excel-independent-review.json')
    ap.add_argument('--sports-a1-evidence-dir', type=Path, default=Path('/home/addaswsw/recommend-support/grid-a2-fd-recovery-20261018/reviews/sports_a1/raw/ga-20261017-sportsa1-train-v1/output'), help='Existing local A1 original receipts; no server query')
    args = ap.parse_args()
    previous = None
    if args.report.exists():
        payload = args.report.read_bytes()
        old = json.loads(payload)
        previous = {'receipt_sha256': digest(payload), 'workbook_sha256': old.get('workbook', {}).get('sha256'), 'status': old.get('status')}
    try:
        report = check(root, args.workbook, args.sports_a1_evidence_dir)
    except Exception as exc:
        report = {'schema': 'grid.excel.nonauthor.review.v1', 'status': 'EXCEL_INDEPENDENT_REVIEW_FAILED', 'reviewer': '/root/fd_repair_reviewer', 'error': type(exc).__name__ + ': ' + str(exc), 'workbook': {'path': str(args.workbook), 'sha256': digest(args.workbook.read_bytes())}, 'review_implementation_sha256': digest(here.read_bytes()), 'workbook_modified': False}
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        raise
    if previous:
        report['supersedes_local_receipt'] = previous
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'workbook': report['workbook'], 'counts': report['counts'], 'report': str(args.report.resolve()), 'report_sha256': digest(args.report.read_bytes()), 'remaining': report['limitations'][-1]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
