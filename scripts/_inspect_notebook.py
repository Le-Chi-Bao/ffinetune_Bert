import json, sys
nb = json.loads(open(sys.argv[1], encoding='utf-8').read())
cells = nb['cells']
print(f'total={len(cells)}')
for i, c in enumerate(cells):
    src = ''.join(c['source'])[:70].replace('\n', ' / ')
    print(f'  cell {i:>2} ({c["cell_type"]}): {src}...')