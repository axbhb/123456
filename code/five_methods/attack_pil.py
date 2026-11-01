from roi_common import *
start=time.time();out=O/'pil';out.mkdir(parents=True,exist_ok=True);state={'method':'PIL-ROI-transfer-v1','status':'starting','pretrain_epochs':30,'attack_steps':30,'epsilon':8/255,'alpha':8/2550,'lambda':.9,'classifier':'official LinearModel, two linear factors, no nonlinear activation','source':'https://github.com/jinlinll/pil','test_data_used':False}
def save(**kw):state.update(kw);state['seconds']=time.time()-start;atomic(out/'attack_status.json',state)
try:
 seed_all();torch.set_num_threads(1);x,y,records=roi_dataset();loader=DataLoader(TensorDataset(x,y),batch_size=512,shuffle=False,num_workers=0)
 cls=module_file('pil_linear',B/'upstream/pil/models/linearmodel.py').LinearModel;model=cls(num_classes=10).cuda();optimizer=torch.optim.SGD(model.parameters(),lr=.003,momentum=.9,weight_decay=0);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30);training=[]
 for epoch in range(30):
  total=correct=0;loss_sum=0;model.train()
  for bx,by in loader:
   bx=bx.cuda();by=by.cuda();optimizer.zero_grad();scores=model(bx);loss=F.cross_entropy(scores,by);loss.backward();optimizer.step();loss_sum+=float(loss)*len(by);correct+=int((scores.argmax(1)==by).sum());total+=len(by)
  scheduler.step();training.append({'epoch':epoch+1,'loss':loss_sum/total,'train_accuracy':correct/total});save(status='linear_pretraining',completed_epochs=epoch+1,training=training)
 torch.save(model.cpu().state_dict(),out/'linear_model.pth');model.cuda().eval();attack=module_file('official_pil_attack',B/'runtime/pil/attacks.py').PILAdversary(model,num_classes=10,epsilon=8/255,alpha=8/2550,lmd=.9,steps=30,device='cuda:0');deltas=torch.zeros_like(x)
 indices=[i for i,r in enumerate(records) if r['selected']]
 for begin in range(0,len(indices),128):
  ind=indices[begin:begin+128];clean=x[ind].cuda();perturbed=attack.attack(clean,y[ind].cuda());deltas[ind]=(perturbed-clean).detach().cpu();save(status='constructing_selected_rois',completed_crops=min(begin+128,len(indices)),total_crops=len(indices))
 torch.save({'selected_indices':indices,'deltas':deltas[indices]},out/'roi_deltas.pt');state=export_roi('pil',deltas,records,state);save(peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3)
except Exception as e:save(status='failed',error=str(e),traceback=traceback.format_exc());raise
