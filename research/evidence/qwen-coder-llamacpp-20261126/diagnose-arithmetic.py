import sys,time,json,subprocess,os,pathlib
sys.path.insert(0,'/workspace/coder')
import benchmark_coder_backends as b
root=pathlib.Path('/workspace/coder');out=root/'evidence'/'diagnostic-arithmetic';out.mkdir(exist_ok=True)
os.environ['LD_LIBRARY_PATH']=str(root/'llama/llama-b12432')+':'+str(root/'llama/cudart-llama-b12432-bin-ubuntu-cuda-12.8-x64')
p=subprocess.Popen([str(root/'llama/llama-b12432/llama-server'),'-m',str(root/'model.gguf'),'--device','CUDA0','-ngl','99','--fit','off','--no-cache-prompt','-c','4096','-np','1','-fa','on','--jinja','--host','127.0.0.1','--port','8799','--alias',b.MODEL,'--api-key',b.KEY],stdout=(out/'engine.log').open('w'),stderr=subprocess.STDOUT)
try:
 for _ in range(60):
  try:
   import urllib.request
   urllib.request.urlopen('http://127.0.0.1:8799/health',timeout=1);break
  except Exception:time.sleep(1)
 prompts=['What does this Python expression return? sum(x*x for x in [2, 3, 5]). Return only the integer.', 'Calculate 4 + 9 + 25. Return only the integer.', 'What is the sum of the squares of 2, 3 and 5? Return only the integer.', 'Explain step by step what Python evaluates for sum(x*x for x in [2, 3, 5]). End with the integer result.', 'Evaluate this Python code and return only the printed integer:\nvalues = [2, 3, 5]\nresult = sum(value ** 2 for value in values)\nprint(result)', 'Return the exact string: KAA-CHECK-38']
 for i,q in enumerate(prompts):
  r=b.request('http://127.0.0.1:8799',b.payload([{'role':'user','content':q}],stream=True))
  with (out/'trials.jsonl').open('a') as f:f.write(json.dumps({'prompt':q,'response':r})+'\n')
finally:
 p.terminate();p.wait(timeout=30)
