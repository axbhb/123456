from roi_common import *
from types import SimpleNamespace
import torchvision
sys.path[:0]=[str(B/'deps'),str(B/'runtime/bait')]
import higher
from toolbox import PerturbationTool
start=time.time();out=O/'bait';out.mkdir(parents=True,exist_ok=True);state={'method':'BAIT-ROI-transfer-v1','status':'starting','epsilon':8/255,'source':'https://github.com/zhli-cs/BAIT','surrogate_pretraining_epochs':3,'outer_rounds':3,'stages':['hard_negative','random_incorrect','most_dissimilar'],'pilot_note':'one outer round per stage; shortened attack optimization, full downstream finetuning retained','test_data_used':False,'surrogate':'ImageNet pretrained ResNet18, nuScenes10 head','class_balancing':'each outer batch contains every nuScenes class; 130 crops (13 per class), deterministic resampling','class_space':'nuScenes10; CIFAR10 flag only selects ten-class branch in official toolbox'}
def save(**kw):state.update(kw);state['seconds']=time.time()-start;atomic(out/'attack_status.json',state)
try:
 seed_all();torch.set_num_threads(1);x,y,records=roi_dataset();net=torchvision.models.resnet18(pretrained=False);net.load_state_dict(torch.load(str(W/'resnet18-f37072fd.pth'),map_location='cpu'));net.fc=torch.nn.Linear(net.fc.in_features,10)
 class Normalized(torch.nn.Module):
  def __init__(self,net):
   super().__init__();self.net=net;self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
  def forward(self,z):return self.net((z-self.mean)/self.std)
 model=Normalized(net).cuda();optimizer=torch.optim.SGD(model.parameters(),lr=.001,momentum=.9,weight_decay=0);criterion=torch.nn.CrossEntropyLoss();loader=DataLoader(TensorDataset(x,y),batch_size=128,shuffle=True,num_workers=0)
 for epoch in range(3):
  total=0;loss_sum=0;model.train()
  for bx,by in loader:
   bx=bx.cuda();by=by.cuda();optimizer.zero_grad();loss=criterion(model(bx),by);loss.backward();optimizer.step();loss_sum+=float(loss)*len(by);total+=len(by)
  save(status='adapting_pretrained_surrogate',completed_epochs=epoch+1,last_loss=loss_sum/total)
 args=SimpleNamespace(noise_shape=[10,3,32,32],epsilon=8/255,train_data_type='CIFAR10');tool=PerturbationTool(args,num_steps=1);class_noise=torch.zeros(10,3,32,32,device='cuda');pools=[torch.where(y==k)[0] for k in range(10)];rng=torch.Generator().manual_seed(0)
 def balanced_indices():
  ind=torch.cat([pool[torch.randint(len(pool),(13,),generator=rng)] for pool in pools]);return ind[torch.randperm(len(ind),generator=rng)]
 functions=[tool.stage1_min_min_attack,tool.stage2_min_min_attack,tool.stage3_min_min_attack];rounds=[]
 for phase,fn in enumerate(functions):
  ind=balanced_indices();inner_x=x[ind].cuda();inner_y=y[ind].cuda();model.train();accum=torch.zeros_like(class_noise);counts=torch.zeros(10,device='cuda')
  with higher.innerloop_ctx(model,optimizer,copy_initial_weights=True) as (copy_model,diff_optimizer):
   inner_loss=criterion(copy_model((inner_x+class_noise[inner_y]).clamp(0,1)),inner_y);diff_optimizer.step(inner_loss)
   for batch_index in range((len(x)+129)//130):
    ind=balanced_indices();bx=x[ind].cuda();by=y[ind].cuda();noise,loss=fn(args,bx,by,copy_model,criterion)
    if not torch.isfinite(noise).all() or not torch.isfinite(loss):raise FloatingPointError('nonfinite BAIT outer step')
    for k in range(10):
     chosen=noise[by==k];accum[k]+=chosen.sum(0).detach();counts[k]+=len(chosen)
    save(status='bilevel_optimization',stage=state['stages'][phase],completed_outer_batches=batch_index+1,total_outer_batches=(len(x)+129)//130,last_outer_loss=float(loss.detach()))
  class_noise=(accum/counts[:,None,None,None]).clamp(-8/255,8/255).detach();optimizer.zero_grad();loss=criterion(model((inner_x+class_noise[inner_y]).clamp(0,1)),inner_y);loss.backward();optimizer.step();rounds.append({'stage':state['stages'][phase],'inner_loss':float(inner_loss.detach()),'last_outer_loss':float(loss.detach()),'class_counts':counts.tolist()})
 torch.save({'generator':tool.generator.state_dict(),'class_noise':class_noise.cpu(),'surrogate':model.state_dict()},out/'bait_surrogate_and_noise.pth');deltas=class_noise.detach().cpu()[y];state['rounds']=rounds;state=export_roi('bait',deltas,records,state);save(peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc());raise
