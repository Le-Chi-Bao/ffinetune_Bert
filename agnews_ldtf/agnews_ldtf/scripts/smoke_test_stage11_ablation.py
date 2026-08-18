"""Stage-11 synthetic architecture, gradient, runner, and aggregation tests."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation_diagnostics import compute_routing_diagnostics
from model_factory import build_model, build_optimizer, extract_logits, validate_optimizer_groups
from training_utils import atomic_torch_save, load_torch_checkpoint, set_seed

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VARIANTS=('A0_full','A1_no_token_router','A2_no_depth_router','A3_final_layer','A4_shared_token_query','A5_shared_depth_query','A6_class_specific_scorer')


def check(value, message):
    if not value: raise AssertionError(message)
def batch():
    ids=torch.tensor([[101,200,201,102,0,0],[101,300,301,302,102,0]],device=DEVICE); return ids,(ids!=0).long(),torch.tensor([0,3],device=DEVICE)
def expected_modules(v):
    return {'A0_full':{'label_query_bank','token_router','depth_router','class_scorer'},'A1_no_token_router':{'label_query_bank','depth_router','class_scorer'},'A2_no_depth_router':{'label_query_bank','token_router','class_scorer'},'A3_final_layer':{'label_query_bank','token_router','class_scorer'},'A4_shared_token_query':{'label_query_bank','shared_token_router','depth_router','class_scorer'},'A5_shared_depth_query':{'label_query_bank','token_router','shared_depth_router','class_scorer'},'A6_class_specific_scorer':{'label_query_bank','token_router','depth_router','class_specific_scorer'}}[v]
def build(v,regime,dim=16,exclude=True):
    return build_model('ldtf_ablation',model_name='bert-base-uncased',num_classes=4,variant=v,token_router_dim=dim,depth_router_dim=dim,exclude_special_tokens=exclude,training_regime=regime).to(DEVICE)
def assert_gradients(model,v,regime):
    # No allowlist: every trainable parameter must participate in the forward pass.
    check(not [n for n,_ in model.named_parameters() if '.pooler.' in n],f'{v}/{regime}: pooler parameter registered')
    check(not [k for k in model.state_dict() if '.pooler.' in k],f'{v}/{regime}: pooler key in state_dict')
    names={n for n,p in model.named_parameters() if p.requires_grad and p.grad is None}
    check(not names,f'{v}/{regime}: unused trainable params with no grad: {sorted(names)}')
    if regime=='frozen': check(all(p.grad is None for p in model.backbone.parameters()),f'{v}: frozen backbone grad')
    else: check(any(p.grad is not None for p in model.backbone.parameters()),f'{v}: no backbone gradient')
def test_variant(v,regime,dim=16,exclude=True):
    set_seed(42); model=build(v,regime,dim,exclude); ids,mask,labels=batch(); model.train(); model.backbone.eval() if regime=='frozen' else None
    before=next(model.backbone.parameters()).detach().clone(); out=model(ids,mask,return_routing=True,return_features=True); logits=extract_logits(out,4); loss=F.cross_entropy(logits,labels)
    opt=build_optimizer(model,training_regime=regime,backbone_learning_rate=2e-5,head_learning_rate=1e-3,weight_decay=.01); validate_optimizer_groups(model,opt,training_regime=regime); loss.backward(); assert_gradients(model,v,regime); opt.step()
    check(logits.shape==(2,4) and torch.isfinite(logits).all() and torch.isfinite(loss),f'{v}: output/loss invalid')
    if regime=='finetune':check(not torch.equal(before,next(model.backbone.parameters()).detach()),f'{v}: backbone did not update')
    registered={name for name,_ in model.named_children() if name!='backbone'};check(registered==expected_modules(v),f'{v}: modules {registered}')
    ta,da=out['token_attention'],out['depth_attention']; valid=(mask.bool() & ~((ids==101)|(ids==102))) if exclude else mask.bool(); invalid=~valid
    check(torch.all(ta.masked_select(invalid[:,None,None,:].expand_as(ta))==0),f'{v}: invalid attention nonzero');check(torch.allclose(ta.sum(-1),torch.ones_like(ta.sum(-1)),atol=1e-5),f'{v}: token normalization');check(torch.allclose(da.sum(-1),torch.ones_like(da.sum(-1)),atol=1e-5),f'{v}: depth normalization')
    if v=='A1_no_token_router':
        check(not any('token_router' in n for n,_ in model.named_parameters()),'A1 token router parameter');check(torch.allclose(ta,ta[:,0:1].expand_as(ta)),'A1 attention not uniform classwise');check(model.label_query_bank.queries.grad is not None and any(p.grad is not None for p in model.depth_router.parameters()),'A1 expected gradients')
    if v=='A2_no_depth_router':check(torch.allclose(da,torch.full_like(da,1/da.shape[-1])), 'A2 depth not uniform');check(torch.allclose(out['fused_features'],out['token_features'].mean(2)),'A2 fused not mean');check(model.label_query_bank.queries.grad is not None and any(p.grad is not None for p in model.token_router.parameters()),'A2 gradients')
    if v=='A3_final_layer':check(da.shape[-1]==1 and torch.all(da==1),'A3 depth');check(torch.allclose(out['token_features'][:,:,0],torch.einsum('bct, btd -> bcd',ta[:,:,0],out['hidden_states'][:,-1])),'A3 used non-final layers')
    if v=='A4_shared_token_query':check(model.shared_token_router.shared_token_query.shape==(1,model.hidden_size),'A4 query');check(torch.allclose(ta,ta[:,0:1].expand_as(ta)),'A4 token attention differs classes');check(model.label_query_bank.queries.grad is not None,'A4 query bank grad')
    if v=='A5_shared_depth_query':check(model.shared_depth_router.shared_depth_query.shape==(1,model.hidden_size),'A5 query');check(model.label_query_bank.queries.grad is not None and model.shared_depth_router.shared_depth_query.grad is not None,'A5 gradients')
    if v=='A6_class_specific_scorer':check(model.class_specific_scorer.class_weights.shape==(4,model.hidden_size) and model.class_specific_scorer.class_bias.shape==(4,),'A6 shapes');check(not hasattr(model,'class_scorer'),'A6 shared scorer registered')
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'x.pt';atomic_torch_save({'model_state_dict':model.state_dict()},p);loaded=build(v,regime,dim,exclude);loaded.load_state_dict(load_torch_checkpoint(p,DEVICE)['model_state_dict'])
    print(f'PASS {v}/{regime}')
def secondary():
    for dim in (64,128):
        model=build('A0_full','frozen',dim);check(model.token_router.query_projection.weight.shape==(dim,model.hidden_size),f'B dim {dim} token shape');check(model.depth_router.query_projection.weight.shape==(dim,model.hidden_size),f'B dim {dim} depth shape');test_variant('A0_full','frozen',dim);print(f'PASS B{dim}')
    ids,mask,_=batch(); model=build('A0_full','frozen',16,True);out=model(ids,mask,return_routing=True,return_features=True);ta=out['token_attention'];special=((ids==101)|(ids==102));check(torch.all(ta.masked_select(special[:,None,None,:].expand_as(ta))==0),'B4 special attention');
    try:model(torch.tensor([[101,102,0]],device=DEVICE),torch.tensor([[1,1,0]],device=DEVICE))
    except ValueError:pass
    else:raise AssertionError('B4 absent-content case did not raise')
    diag=compute_routing_diagnostics(ta,out['depth_attention'],out['label_queries'],ids,mask);check(len(diag['mean_depth_weight_by_layer'])==12 and len(diag['query_cosine_similarity'])==4,'diagnostics malformed');print('PASS B4 and diagnostics')
def fixture_runner():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); base=root/'stage10';base.mkdir();base_config={'epochs':1,'train_batch_size':2,'eval_batch_size':2,'gradient_accumulation_steps':1,'backbone_learning_rate':2e-5,'head_learning_rate':1e-3,'weight_decay':.01,'warmup_ratio':.1,'max_grad_norm':1.,'mixed_precision':'no','early_stopping_patience':1,'num_workers':0,'max_length':16};(base/'config.json').write_text(json.dumps(base_config));(base/'summary.json').write_text(json.dumps({'training_regime':'frozen','official_test_evaluated':False}))
        cfg={'seed':42,'base_model':'bert-base-uncased','training_regime':'from_stage10','train_path':'never-read','validation_path':'never-read','variants':[{'name':'ok','variant':'A0_full'},{'name':'bad','variant':'A1_no_token_router'}]};cp=root/'config.json';cp.write_text(json.dumps(cfg))
        fake=root/'fake_train.py';fake.write_text("import sys,json\nfrom pathlib import Path\na=sys.argv; run=Path(a[a.index('--output-dir')+1])/a[a.index('--run-name')+1]\nif a[a.index('--run-name')+1]=='bad': raise SystemExit(7)\n(run/'checkpoints').mkdir(parents=True,exist_ok=True); (run/'checkpoints'/'last.pt').write_text('x'); (run/'summary.json').write_text(json.dumps({'official_test_evaluated':False}))\n")
        out=root/'out'; cmd=[sys.executable,'-m','scripts.run_stage11_ablation','--config',str(cp),'--base-stage10-run',str(base),'--output-root',str(out),'--train-entrypoint',str(fake),'--continue-on-error'];subprocess.run(cmd,check=True);reg=json.loads((out/'run_registry.json').read_text());check([x['status'] for x in reg]==['PASS','FAIL'],'runner continue/failure registry');check(len(json.loads((out/'failures.json').read_text()))==1,'runner failure json')
        subprocess.run(cmd+['--skip-completed'],check=True);reg=json.loads((out/'run_registry.json').read_text());check(reg[0]['status']=='SKIPPED_COMPLETED','runner skip completed')
        (out/'ok'/'summary.json').unlink();subprocess.run(cmd+['--skip-completed'],check=True);reg=json.loads((out/'run_registry.json').read_text());check(reg[0].get('resume_detected') is True,'runner resume detection')
        bad_cfg=dict(cfg);bad_cfg['training_regime']='finetune';cp.write_text(json.dumps(bad_cfg));result=subprocess.run(cmd,capture_output=True,text=True);check(result.returncode!=0,'runner conflicting regime not rejected');print('PASS runner fixture')

def fixture_aggregate():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); sig={'x':1}; reg=[]
        for name,f1,params in [('A0_full',.8,100),('A1_no_token_router',.7,90)]:
            run=root/name;(run/'metrics').mkdir(parents=True);reg.append({'name':name,'status':'PASS','path':str(run)}); (run/'summary.json').write_text(json.dumps({'official_test_evaluated':False,'best_validation_loss':1,'best_validation_accuracy':.8,'best_validation_macro_f1':f1,'total_parameters':params,'trainable_parameters':params,'peak_vram_mb':1,'training_time_seconds':1}));(run/'config.json').write_text(json.dumps({'seed':42,'training_regime':'frozen','effective_batch_size':8,'max_length':128,'warmup_ratio':.1,'weight_decay':.01}));(run/'data_signature.json').write_text(json.dumps(sig));(run/'ablation_config.json').write_text(json.dumps({'variant':name,'token_router_dim':256}));            (run/'metrics'/'best_validation_metrics.json').write_text(
                json.dumps({'per_class': {c: {'f1': .5} for c in ['World', 'Sports', 'Business', 'Sci/Tech']}})
            )
        (root/'run_registry.json').write_text(json.dumps(reg));out=root/'r.csv';js=root/'r.json';cmd=[sys.executable,'-m','scripts.aggregate_stage11_ablation','--root',str(root),'--output-csv',str(out),'--output-json',str(js)];subprocess.run(cmd,check=True);r=json.loads(js.read_text())['results'];check(abs(r[1]['Delta F1 vs A0']+.1)<1e-9 and abs(r[1]['Macro F1 drop vs A0']-.1)<1e-9 and r[1]['parameter_delta_vs_full']==-10,'aggregate deltas');check(out.is_file() and js.is_file(),'aggregate outputs')
        cases=[('data_signature.json',{'x':2}),('config.json',{'seed':7,'training_regime':'frozen','effective_batch_size':8,'max_length':128,'warmup_ratio':.1,'weight_decay':.01}),('config.json',{'seed':42,'training_regime':'finetune','effective_batch_size':8,'max_length':128,'warmup_ratio':.1,'weight_decay':.01}),('summary.json',{'official_test_evaluated':True,'best_validation_loss':1,'best_validation_accuracy':.8,'best_validation_macro_f1':.7,'total_parameters':90,'trainable_parameters':90,'peak_vram_mb':1,'training_time_seconds':1})]
        target=root/'A1_no_token_router'
        for filename,replacement in cases:
            original=(target/filename).read_text();(target/filename).write_text(json.dumps(replacement));result=subprocess.run(cmd,capture_output=True,text=True);check(result.returncode!=0,f'aggregate did not reject {filename}');(target/filename).write_text(original)
        print('PASS aggregator fixture')
def pooler_removal():
    """Verify the pooler is gone and the parameter delta matches 768*768+768."""
    expected_removed=768*768+768
    model=build('A0_full','finetune',16)
    check(getattr(model.backbone.bert,'pooler',None) is None,'backbone still exposes a pooler')
    check(not [n for n,_ in model.named_parameters() if '.pooler.' in n],'pooler parameters still registered')
    check(not [k for k in model.state_dict() if '.pooler.' in k],'pooler keys still in state_dict')
    total=sum(p.numel() for p in model.parameters())
    check(total==109535233-expected_removed,f'unexpected total parameter count {total}')
    check(all(p.requires_grad for p in model.backbone.parameters()),'finetune backbone not fully trainable')
    print(f'PASS pooler removal (removed {expected_removed} parameters; total now {total})')

def main():
    pooler_removal()
    for v in VARIANTS:
        for r in ('frozen','finetune'):test_variant(v,r)
    secondary();fixture_runner();fixture_aggregate();print('All Stage-11 implementation tests PASSED. Official test was not loaded or evaluated.')
if __name__=='__main__':main()
