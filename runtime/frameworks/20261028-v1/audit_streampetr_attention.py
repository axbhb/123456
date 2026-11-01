"""CPU audit of learned cross-attention parameter compatibility, not a detector run."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,json,time,hashlib,math
from pathlib import Path
import torch
import torch.nn.functional as F
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();started=time.time();torch.set_num_threads(1)
assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()=='e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451'
try:payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
except TypeError:payload=torch.load(a.checkpoint,map_location='cpu')
sd=payload['state_dict'];rows=[];torch.manual_seed(7)
for layer in range(6):
 prefix='pts_bbox_head.transformer.decoder.layers.%d.attentions.1.attn.'%layer
 state={k[len(prefix):]:v for k,v in sd.items() if k.startswith(prefix)}
 m=torch.nn.MultiheadAttention(256,8,dropout=0.0);m.load_state_dict(state,strict=True);m.eval()
 q=torch.randn(5,2,256,requires_grad=True);k=torch.randn(9,2,256,requires_grad=True);v=torch.randn(9,2,256,requires_grad=True)
 actual=m(q,k,v,need_weights=False)[0]
 wq,wk,wv=m.in_proj_weight.chunk(3);bq,bk,bv=m.in_proj_bias.chunk(3)
 def project(x,w,b):return F.linear(x,w,b).permute(1,0,2).reshape(2,-1,8,32).transpose(1,2)
 pq,pk,pv=project(q,wq,bq),project(k,wk,bk),project(v,wv,bv)
 soft=torch.softmax(pq@pk.transpose(-1,-2)/math.sqrt(32),dim=-1)
 reference=F.linear((soft@pv).transpose(1,2).reshape(2,5,256).transpose(0,1),m.out_proj.weight,m.out_proj.bias)
 error=(actual-reference).abs().max().item();torch.testing.assert_close(actual,reference,rtol=1e-5,atol=2e-6)
 ga=torch.autograd.grad(actual.square().mean(),(q,k,v),retain_graph=True);gr=torch.autograd.grad(reference.square().mean(),(q,k,v))
 for x,y in zip(ga,gr):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-7)
 rows.append({'layer':layer,'strict_parameter_load':True,'max_abs_forward_error':error,'max_abs_input_gradient_error':max((x-y).abs().max().item() for x,y in zip(ga,gr))})
assert not torch.cuda.is_initialized()
r={'status':'passed','scope':'official pretrained cross-attention weights into torch MultiheadAttention, compared with explicit scaled dot-product reference on synthetic features; no detector inference','layers':rows,'gpu_used':False,'full_detector_verified':False,'seconds':time.time()-started}
a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r))
