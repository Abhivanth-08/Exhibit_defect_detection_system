import re, pathlib
p = pathlib.Path('app2.py')
src = p.read_text(encoding='utf-8')
src = src.replace('use_container_width=True', 'width="stretch"')
src = src.replace('use_container_width=False', 'width="content"')
p.write_text(src, encoding='utf-8')
print('Fixed. stretch count:', src.count('width="stretch"'))
