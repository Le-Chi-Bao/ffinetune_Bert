"""Aggregate compatible validation-only Stage-11 ablation runs."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def load(path):
    with open(path) as f:return json.load(f)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',default='A0_full');p.add_argument('--output-csv',required=True);p.add_argument('--output-json',required=True);a=p.parse_args();root=Path(a.root); registry=load(root/'run_registry.json'); rows=[]; reference=None; signature=None; locked=None
 for entry in registry:
  if entry['status'] not in {'PASS','SKIPPED_COMPLETED'}:continue
  run=Path(entry['path']); summary=load(run/'summary.json');config=load(run/'config.json');sig=load(run/'data_signature.json');metrics=load(run/'metrics'/'best_validation_metrics.json');ablation=load(run/'ablation_config.json')
  if summary.get('official_test_evaluated') is not False: raise ValueError(f'{run}: official test was evaluated')
  comparability={'seed':config['seed'],'training_regime':config['training_regime'],'effective_batch_size':config['effective_batch_size'],'max_length':config['max_length'],'warmup_ratio':config['warmup_ratio'],'weight_decay':config['weight_decay'],'checkpoint_metric':'f1_macro'}
  if signature is None:signature=sig;locked=comparability
  if sig!=signature or comparability!=locked:raise ValueError(f'Incompatible comparison: {run}')
  pc=metrics['per_class']; row={'Variant':entry['name'],'Token routing':'none' if ablation['variant']=='A1_no_token_router' else ('shared' if ablation['variant']=='A4_shared_token_query' else 'label-conditioned'),'Depth routing':'none' if ablation['variant'] in {'A2_no_depth_router','A3_final_layer'} else ('shared-query' if ablation['variant']=='A5_shared_depth_query' else 'label-conditioned'),'Layer mode':'final' if ablation['variant']=='A3_final_layer' else 'all_layers','Scorer':'class-specific' if ablation['variant']=='A6_class_specific_scorer' else 'shared','Router dim':ablation.get('token_router_dim',256),'Val loss':summary['best_validation_loss'],'Accuracy':summary['best_validation_accuracy'],'Macro F1':summary['best_validation_macro_f1'],'Params':summary['total_parameters'],'Trainable params':summary['trainable_parameters'],'Peak VRAM MB':summary['peak_vram_mb'],'Time seconds':summary['training_time_seconds'],**{f'{c} F1':pc[c]['f1'] for c in pc}}
  rows.append(row)
  if entry['name']==a.reference:reference=row
 if reference is None:raise ValueError(f'Reference {a.reference} not found')
 for row in rows:row['Delta F1 vs A0']=row['Macro F1']-reference['Macro F1'];row['Macro F1 drop vs A0']=reference['Macro F1']-row['Macro F1'];row['parameter_delta_vs_full']=row['Params']-reference['Params'];row['trainable_parameter_delta_vs_full']=row['Trainable params']-reference['Trainable params']
 for out in (Path(a.output_csv),Path(a.output_json)):out.parent.mkdir(parents=True,exist_ok=True)
 with open(a.output_csv,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 Path(a.output_json).write_text(json.dumps({'comparison_protocol':locked,'data_signature':signature,'results':rows,'note':'single-seed provisional ablation results; no statistical significance claim'},indent=2)+'\n')
 with open(root/'ablation_effects.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=['Variant','Delta F1 vs A0','Macro F1 drop vs A0','parameter_delta_vs_full','trainable_parameter_delta_vs_full']);w.writeheader();w.writerows([{k:r[k] for k in w.fieldnames} for r in rows])
 print(f'Aggregated {len(rows)} compatible validation-only variants.')
if __name__=='__main__':main()
