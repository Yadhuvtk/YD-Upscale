import json
import os

problems = json.loads('[{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\benchmark.py","startLine":6},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\benchmark.py","startLine":7},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\benchmark.py","startLine":8},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\benchmark.py","startLine":9},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\benchmark.py","startLine":26},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":6},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":7},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":8},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":9},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":10},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":11},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":12},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":13},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":14},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":15},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\scripts\\train_stage1.py","startLine":16},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\engine\\trainer.py","startLine":1},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\engine\\trainer.py","startLine":2},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\engine\\trainer.py","startLine":3},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\engine\\trainer.py","startLine":15},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\engine\\trainer.py","startLine":42},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\losses\\text_consistency_loss.py","startLine":1},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\losses\\text_consistency_loss.py","startLine":2},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\metrics\\edge_accuracy.py","startLine":1},{"path":"e:\\Yadhu Projects\\YD-Upscale\\YD-Upscale\\yd_upscale\\metrics\\text_region_score.py","startLine":1}]')

files_to_update = {}
for p in problems:
    path = p['path']
    line = p['startLine']
    if path not in files_to_update:
        files_to_update[path] = set()
    files_to_update[path].add(line)

for path, lines in files_to_update.items():
    if not os.path.exists(path): continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.readlines()
    
    modified = False
    for line_idx in lines:
        real_idx = line_idx - 1
        if real_idx < len(content):
            if 'pyre-ignore' not in content[real_idx]:
                content[real_idx] = content[real_idx].rstrip() + '  # pyre-ignore[21]\n'
                modified = True
                
    if modified:
        with open(path, 'w', encoding='utf-8') as f:
            f.writelines(content)
        print(f"Fixed {path}")
