"""Run Stage-11 variants sequentially through shared, validation-only train.py."""
from __future__ import annotations
import argparse, json, subprocess, sys, traceback
from pathlib import Path
from typing import Any

def load(path: Path) -> dict[str, Any]:
    with path.open(encoding='utf-8') as handle:return json.load(handle)
def save(value: Any,path: Path)->None:path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def resolve_regime(config: dict[str,Any], summary: dict[str,Any])->str:
    actual=summary.get('training_regime'); requested=config.get('training_regime','from_stage10')
    if actual not in {'frozen','finetune'}:raise RuntimeError('Stage-10 summary has no valid provisional regime.')
    if requested not in {'from_stage10',actual}:raise RuntimeError(f'Stage-11 config regime={requested!r} conflicts with Stage-10 regime={actual!r}.')
    return actual
def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--base-stage10-run',required=True);p.add_argument('--output-root',required=True);p.add_argument('--train-entrypoint',default='train.py');p.add_argument('--skip-completed',action='store_true');p.add_argument('--continue-on-error',action='store_true');a=p.parse_args()
    cfg=load(Path(a.config));base=Path(a.base_stage10_run);base_config=load(base/'config.json');summary=load(base/'summary.json')
    if summary.get('official_test_evaluated') is not False:raise RuntimeError('Base Stage-10 run evaluated official test.')
    regime=resolve_regime(cfg,summary);root=Path(a.output_root);root.mkdir(parents=True,exist_ok=True);registry=[];failures=[]
    for variant in cfg['variants']:
        name=variant['name'];run=root/name;completed=run/'summary.json';entry={'name':name,'path':str(run),'training_regime':regime}
        if a.skip_completed and completed.is_file() and load(completed).get('official_test_evaluated') is False:
            entry['status']='SKIPPED_COMPLETED';registry.append(entry);continue
        command=[sys.executable,a.train_entrypoint,'--model-type','ldtf_ablation','--ablation-variant',variant['variant'],'--training-regime',regime,'--model-name',cfg['base_model'],'--num-classes','4','--train-path',cfg['train_path'],'--validation-path',cfg['validation_path'],'--output-dir',str(root),'--run-name',name]
        mapping={'seed':'seed','epochs':'epochs','train_batch_size':'train-batch-size','eval_batch_size':'eval-batch-size','gradient_accumulation_steps':'gradient-accumulation-steps','backbone_learning_rate':'backbone-learning-rate','head_learning_rate':'head-learning-rate','weight_decay':'weight-decay','warmup_ratio':'warmup-ratio','max_grad_norm':'max-grad-norm','mixed_precision':'mixed-precision','early_stopping_patience':'early-stopping-patience','num_workers':'num-workers','max_length':'max-length'}
        for key,flag in mapping.items():command += [f'--{flag}',str(base_config[key] if key!='seed' else cfg['seed'])]
        command += ['--token-router-dim',str(variant.get('token_router_dim',256)),'--depth-router-dim',str(variant.get('depth_router_dim',256))]
        if variant.get('exclude_special_tokens'):command.append('--exclude-special-tokens')
        if (run/'checkpoints'/'last.pt').is_file():command += ['--resume-from',str(run/'checkpoints'/'last.pt')];entry['resume_detected']=True
        try:
            subprocess.run(command,check=True)
            resolved=dict(variant);resolved['training_regime']=regime;resolved['base_stage10_run']=str(base.resolve());save(resolved,run/'ablation_config.json');entry['status']='PASS'
        except Exception as error:
            entry.update({'status':'FAIL','error':str(error),'traceback':traceback.format_exc()});failures.append(entry)
            if not a.continue_on_error:registry.append(entry);break
        registry.append(entry)
    save(registry,root/'run_registry.json');save(failures,root/'failures.json');print(f'Wrote registry for {len(registry)} variants. failures={len(failures)}')
if __name__=='__main__':main()
