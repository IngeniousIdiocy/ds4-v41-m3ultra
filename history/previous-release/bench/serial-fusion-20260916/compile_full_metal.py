from pathlib import Path
import re,ast
r=Path(__file__).resolve().parents[2];d=Path(__file__).resolve().parent
s=(r/'ds4_metal.m').read_text();base=s.split('static const char *ds4_gpu_source =',1)[1].split('static NSString *ds4_gpu_full_source',1)[0]
text=''.join(ast.literal_eval(x) for x in re.findall(r'"(?:[^"\\]|\\.)*"',base))
files=re.findall(r'@"(metal/[^\"]+)"',s.split('required_sources = @[',1)[1].split('];',1)[0])
for name in files:text+='\n'+(r/name).read_text()
(d/'full-source.metal').write_text(text)
print('Assembled',len(files),'Metal files;',len(text),'bytes')
