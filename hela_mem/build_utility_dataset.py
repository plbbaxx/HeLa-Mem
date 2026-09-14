"""Build a Base-conditioned downstream-intervention memory utility dataset."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, itertools, json, math, os, random, statistics, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from .analyze_associative_candidate_expansion import base_top_k_ids, load_jsonl
from .runtime import atomic_write_json, chat_extra_body, git_commit, model_for, sha256_file, strip_reasoning

EPSILONS=(0.0,0.02,0.05,0.10); SPLIT_SEED=20260914
# Frozen LongMemEval-S protocol exclusions.  Keeping this small immutable value
# local lets offline dataset/statistics helpers be imported without initializing
# the OpenAI-backed evaluation stack.
CORRUPTED_INDICES={74,183,278,351,380}
def reader_prompt(*args,**kwargs):
    from .eval_longmemeval import build_longmemeval_prompt
    return build_longmemeval_prompt(*args,**kwargs)
def judge_prompt(*args,**kwargs):
    from .eval_longmemeval import get_anscheck_prompt
    return get_anscheck_prompt(*args,**kwargs)
def judge_result(raw):
    from .eval_longmemeval import parse_judge_response
    return parse_judge_response(raw)
def dump_jsonl(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')
    os.replace(tmp,path)
def fingerprint(x): return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def cache_read(path, fp):
    try:
        x=json.loads(Path(path).read_text(encoding='utf-8')); return x if x.get('status')=='ok' and x.get('fingerprint')==fp else None
    except Exception:return None
def graph_candidates(item,prediction,graph,top_k=15):
    nodes,edges=graph.get('nodes',{}),graph.get('edges',{}); base=base_top_k_ids(prediction,top_k); base_set=set(base); found={}
    retrieved={str(x.get('node_id')):x for x in prediction.get('retrieved_episodic',[]) if x.get('node_id') is not None}
    for parent in base:
        for cid,w in edges.get(parent,{}).items():
            cid=str(cid)
            if cid in base_set or cid not in nodes: continue
            row=found.setdefault(cid,{'question_id':str(item['question_id']),'candidate_memory_id':cid,'candidate_text':nodes[cid].get('content',''),'session_id':nodes[cid].get('session_id'),'timestamp':nodes[cid].get('timestamp'),'source_edges':[]})
            row['source_edges'].append({'parent_memory_id':parent,'edge_weight':float(w)})
    for cid,row in found.items():
        weights=[x['edge_weight'] for x in row['source_edges']]; saved=retrieved.get(cid,{})
        row.update({'original_hebbian_score':max(weights) if weights else None,'hebbian_score':max(weights) if weights else None,'original_spreading_score':saved.get('score') if saved.get('source')=='hebbian' else None,'spreading_score':saved.get('score') if saved.get('source')=='hebbian' else None,'semantic_score':saved.get('base_score')})
    return base,sorted(found.values(),key=lambda x:(-float(x['hebbian_score'] or 0),x['candidate_memory_id']))
def frozen_side_context(qid,prediction,mem_dir):
    semantic=prediction.get('retrieved_semantic',[]); knowledge='\n'.join(f"- {x.get('knowledge','')}" for x in semantic if x.get('knowledge'))
    profile='None'; assistant=''
    try:
        from .hebbian_knowledge_memory import HebbianKnowledgeMemory
        kb=HebbianKnowledgeMemory(file_path=str(Path(mem_dir)/f'{qid}_long_term.json')); profile=kb.get_raw_user_profile(qid) or 'None'; aks=kb.get_assistant_knowledge() or []
        if aks: assistant='Here are some of your character traits and knowledge:\n'+''.join(f"- {x.get('knowledge','').strip()}\n" for x in aks if x.get('knowledge','').strip())
    except Exception: pass
    return knowledge,profile,assistant
def memory_block(node,source='Direct Match',score=None):
    score=0.0 if score is None else float(score)
    return f"[{source} | Relevancy: {score:.2f}]\nTime: {node.get('timestamp','unknown')}\nContent: {node.get('content','')}"
def build_context_record(item,prediction,graph,mem_dir,top_k):
    base_ids,candidates=graph_candidates(item,prediction,graph,top_k); nodes=graph.get('nodes',{}); byid={str(x.get('node_id')):x for x in prediction.get('retrieved_episodic',[])}
    blocks=[memory_block(nodes[x], 'Direct Match' if float(byid.get(x,{}).get('base_score') or 0)>0.6 else 'Associative Memory',byid.get(x,{}).get('score')) for x in base_ids if x in nodes]
    knowledge,profile,assistant=frozen_side_context(str(item['question_id']),prediction,mem_dir)
    system,user=reader_prompt('\n\n'.join(blocks),knowledge,profile,assistant,item['question'],item.get('question_date',''))
    return {'question_id':str(item['question_id']),'question':item['question'],'question_type':item.get('question_type',''),'question_date':item.get('question_date',''),'gold_answer':str(item.get('answer','')),'base_memory_ids':base_ids,'base_memories':[nodes[x] for x in base_ids if x in nodes],'base_context_text':'\n\n'.join(blocks),'system_prompt':system,'baseline_user_prompt':user,'knowledge_text':knowledge,'profile_text':profile,'assistant_knowledge_text':assistant,'candidates':candidates}
def intervention_messages(base,candidate=None):
    context=base['base_context_text']
    if candidate: context += ('\n\n' if context else '')+memory_block({'timestamp':candidate.get('timestamp'),'content':candidate['candidate_text']},'Associative Memory',candidate.get('spreading_score') or candidate.get('hebbian_score'))
    system,user=reader_prompt(context,base['knowledge_text'],base['profile_text'],base['assistant_knowledge_text'],base['question'],base['question_date'])
    return [{'role':'system','content':system},{'role':'user','content':user}],user,context
class Calls:
    def __init__(self):self.c=Counter()
    def chat(self,messages,role,model,seed=42):
        from openai import OpenAI
        client=OpenAI(api_key=os.environ.get('OPENAI_API_KEY','EMPTY'),base_url=os.environ.get('OPENAI_BASE_URL')); self.c[f'new_{role}_calls']+=1
        r=client.chat.completions.create(model=model,messages=messages,temperature=0.0,top_p=1.0,seed=seed,max_tokens=2000,**chat_extra_body()); return strip_reasoning(r.choices[0].message.content if r.choices else '')
def gold_logprob_values(token_logprobs,start,full_length):
    if len(token_logprobs)<full_length:
        raise RuntimeError(f'incomplete prompt logprobs: expected>={full_length}, received={len(token_logprobs)}, gold_start={start}')
    gold_lps=token_logprobs[start:full_length]
    missing=[start+i for i,value in enumerate(gold_lps) if value is None]
    if missing:
        raise RuntimeError(f'missing gold token logprobs at token positions {missing[:10]} (count={len(missing)})')
    return [float(value) for value in gold_lps]
class GoldScorer:
    def __init__(self,model,path,calls): self.model=model; self.calls=calls; self.available=False; self.reason='not probed'; self.tok=None
    def probe(self,messages,gold):
        try:
            from transformers import AutoTokenizer
            self.tok=AutoTokenizer.from_pretrained(self.path,trust_remote_code=True,local_files_only=True); self.calls.c['logprob_probe_calls']+=1; self.score(messages,gold,probe=True); self.available=True; self.reason='vLLM completions prompt_logprobs available'
        except Exception as e:self.available=False; self.reason=f'{type(e).__name__}: {e}'[:500]
    @property
    def path(self): return os.environ.get('HEBBIAN_LOCAL_MODEL_PATH','')
    def score(self,messages,gold,probe=False):
        from openai import OpenAI
        if not self.tok: raise RuntimeError('tokenizer unavailable')
        prefix=self.tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
        full=self.tok.apply_chat_template(messages+[{'role':'assistant','content':gold}],tokenize=True,continue_final_message=True)
        start=0
        while start<min(len(prefix),len(full)) and prefix[start]==full[start]: start+=1
        if start>=len(full): raise RuntimeError('gold token span empty')
        client=OpenAI(api_key=os.environ.get('OPENAI_API_KEY','EMPTY'),base_url=os.environ.get('OPENAI_BASE_URL')); self.calls.c['new_logprob_calls']+=0 if probe else 1
        # vLLM's prompt-only scoring path is echo=True with max_tokens=0.
        # No sampled continuation is requested or included in utility.
        r=client.completions.create(model=self.model,prompt=full,max_tokens=0,temperature=0.0,top_p=1.0,seed=42,echo=True,logprobs=1,extra_body={'prompt_logprobs':1,'add_special_tokens':False})
        lps=r.choices[0].logprobs.token_logprobs
        vals=gold_logprob_values(lps,start,len(full))
        return {'total_logprob':sum(vals),'mean_logprob_per_token':sum(vals)/len(vals),'token_count':len(vals)}
def cached_call(path,fp,fn,calls,key):
    hit=cache_read(path,fp)
    if hit:calls.c[f'{key}_cache_hits']+=1; return hit
    try: payload=fn(); row={'status':'ok','fingerprint':fp,**payload}; atomic_write_json(path,row); return row
    except Exception as e:calls.c['failed_calls']+=1; row={'status':'error','fingerprint':fp,'error':f'{type(e).__name__}: {e}'[:1000]}; atomic_write_json(path,row); return row
def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]); out=[0.0]*len(values); i=0
    while i<len(order):
        j=i
        while j+1<len(order) and values[order[j+1]]==values[order[i]]:j+=1
        rank=(i+j)/2+1
        for k in range(i,j+1):out[order[k]]=rank
        i=j+1
    return out
def pearson(x,y):
    if len(x)<2:return None
    mx,my=statistics.mean(x),statistics.mean(y); den=math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y)); return sum((a-mx)*(b-my) for a,b in zip(x,y))/den if den else None
def describe(v):
    if not v:return {'count':0}
    q=statistics.quantiles(v,n=100,method='inclusive') if len(v)>1 else [v[0]]*99
    return {'count':len(v),'min':min(v),'max':max(v),'mean':statistics.mean(v),'std':statistics.pstdev(v),'p10':q[9],'p25':q[24],'median':statistics.median(v),'p75':q[74],'p90':q[89]}
def transition(a,b):return ('C' if a else 'W')+'2'+('C' if b else 'W')
def utility_profiles(rows,threshold=.02):
    grouped=defaultdict(list)
    for row in rows:grouped[row['question_id']].append(float(row['utility_score']))
    return {qid:{'has_positive':max(values)>threshold,'has_negative':min(values)<-threshold,'has_within_question_variation':max(values)-min(values)>threshold,'candidate_count':len(values),'utility_min':min(values),'utility_max':max(values)} for qid,values in grouped.items()}
def stratified_split(rows):
    """Deterministic multi-label split over candidate-bearing questions only."""
    profiles=utility_profiles(rows);ids=sorted(profiles);n=len(ids)
    capacities={'train':int(.8*n),'dev':int(.1*n)};capacities['test']=n-capacities['train']-capacities['dev']
    labels=('has_positive','has_negative','has_within_question_variation')
    totals={label:sum(profiles[qid][label] for qid in ids) for label in labels}
    targets={split:{label:totals[label]*capacities[split]/max(1,n) for label in labels} for split in capacities}
    rng=random.Random(SPLIT_SEED);tie={qid:rng.random() for qid in ids}
    rarity=lambda qid:sum(1/max(1,totals[label]) for label in labels if profiles[qid][label])
    order=sorted(ids,key=lambda qid:(-rarity(qid),-profiles[qid]['candidate_count'],tie[qid],qid))
    assigned={split:[] for split in capacities};counts={split:Counter() for split in capacities}
    for qid in order:
        available=[split for split in capacities if len(assigned[split])<capacities[split]]
        def score(split):
            label_need=sum(max(0.0,targets[split][label]-counts[split][label])/max(1.0,targets[split][label]) for label in labels if profiles[qid][label])
            capacity_need=(capacities[split]-len(assigned[split]))/max(1,capacities[split])
            return (label_need,capacity_need,{'test':2,'dev':1,'train':0}[split])
        chosen=max(available,key=score);assigned[chosen].append(qid)
        for label in labels:counts[chosen][label]+=int(profiles[qid][label])
    return {split:sorted(values) for split,values in assigned.items()},profiles
def build_pairs(rows,eps,semantic_lo,semantic_hi,utility_p75):
    grouped=defaultdict(list)
    for r in rows:grouped[r['question_id']].append(r)
    out=[]
    for qid,items in grouped.items():
        for a,b in itertools.combinations(items,2):
            d=a['utility_score']-b['utility_score']
            if abs(d)<=eps:continue
            win,lose=(a,b) if d>0 else (b,a); gap=abs(d); types=[]
            if win['utility_score']>0 and abs(lose['utility_score'])<=.02:types.append('positive-vs-neutral')
            if win['utility_score']>0 and lose['utility_score']<0:types.append('positive-vs-negative')
            types.append('large-utility-gap' if gap>=.10 else 'small-utility-gap')
            if semantic_hi is not None and (lose.get('semantic_score') or -1)>=semantic_hi and gap>=.10:types.append('high-semantic-similarity-but-large-utility-gap')
            if utility_p75 is not None and win['utility_score']>=utility_p75 and win.get('semantic_score') is not None and semantic_lo is not None and win['semantic_score']<=semantic_lo:types.append('low-semantic-similarity-but-strong-positive')
            out.append({'question_id':qid,'preferred_candidate_id':win['candidate_memory_id'],'rejected_candidate_id':lose['candidate_memory_id'],'utility_margin':gap,'epsilon':eps,'pair_types':types})
    return out
def main():
    p=argparse.ArgumentParser();
    for x in ('data-path','mem-dir','base-predictions','oracle-v05','output-dir','local-model-path'):p.add_argument('--'+x,required=True)
    p.add_argument('--top-k',type=int,default=15);p.add_argument('--workers',type=int,default=8);p.add_argument('--num-items',type=int);p.add_argument('--require-continuous',action=argparse.BooleanOptionalAction,default=True);a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);os.environ['HEBBIAN_LOCAL_MODEL_PATH']=a.local_model_path
    data=json.loads(Path(a.data_path).read_text(encoding='utf-8')); data=data[:a.num_items] if a.num_items else data; preds=load_jsonl(Path(a.base_predictions)); calls=Calls(); bases=[]; failures=[]
    for idx,item in enumerate(data):
        qid=str(item['question_id']); gp=Path(a.mem_dir)/f'{qid}_hebbian.json'
        if qid not in preds or not gp.exists(): failures.append({'question_id':qid,'stage':'export','error':'missing prediction or graph'});continue
        base=build_context_record(item,preds[qid],json.loads(gp.read_text(encoding='utf-8')),a.mem_dir,a.top_k);base['dataset_index']=idx;base['is_corrupted']=idx in CORRUPTED_INDICES;bases.append(base)
    dump_jsonl(out/'base_contexts.jsonl',[{k:v for k,v in b.items() if k!='candidates'} for b in bases]); candidates=[(b,c) for b in bases for c in b['candidates']]
    dump_jsonl(out/'candidate_pool.jsonl',[c for _,c in candidates])
    gen_model=model_for('generation');judge_model=model_for('judge'); scorer=GoldScorer(gen_model,a.local_model_path,calls)
    probe_base=next((b for b in bases if b['candidates']),None)
    if probe_base: scorer.probe(intervention_messages(probe_base)[0],probe_base['gold_answer'])
    if a.require_continuous and not scorer.available:raise RuntimeError(f'continuous utility is required but unavailable: {scorer.reason}')
    def baseline(b):
        qid=b['question_id']; messages,user,context=intervention_messages(b); fp=fingerprint({'v':1,'messages':messages,'model':gen_model,'temperature':0,'top_p':1,'seed':42})
        ans=cached_call(out/'cache/baseline_answer'/f'{qid}.json',fp,lambda:{'answer':calls.chat(messages,'generation',gen_model)},calls,'baseline_answer')
        if ans['status']!='ok':return None
        jp=judge_prompt(b['question_type'],b['question'],b['gold_answer'],ans['answer'],abstention='abs' in qid); jfp=fingerprint({'v':1,'prompt':jp,'model':judge_model})
        judged=cached_call(out/'cache/baseline_judge'/f'{qid}.json',jfp,lambda:{'raw':calls.chat([{'role':'user','content':jp}],'judge',judge_model),'correct':None},calls,'baseline_judge');
        if judged['status']=='ok' and judged.get('correct') is None: judged['correct']=judge_result(judged['raw']);atomic_write_json(out/'cache/baseline_judge'/f'{qid}.json',judged)
        lp=None
        if scorer.available:
            lfp=fingerprint({'v':1,'messages':messages,'gold':b['gold_answer'],'model':gen_model});lp=cached_call(out/'cache/baseline_logprob'/f'{qid}.json',lfp,lambda:scorer.score(messages,b['gold_answer']),calls,'baseline_logprob')
        return {'answer':ans,'judge':judged,'logprob':lp,'messages':messages,'context':context}
    base_results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
        fut={ex.submit(baseline,b):b['question_id'] for b in bases if b['candidates']}
        for f in concurrent.futures.as_completed(fut):
            qid=fut[f]
            try:base_results[qid]=f.result()
            except Exception as e:failures.append({'question_id':qid,'stage':'baseline','error':f'{type(e).__name__}: {e}'[:1000]})
            print(f"Baseline: {len(base_results)}/{len(fut)}",flush=True)
    def intervene(task):
        b,c=task;qid=b['question_id'];cid=c['candidate_memory_id'];base=base_results.get(qid)
        if not base or base['judge']['status']!='ok':return None
        messages,user,context=intervention_messages(b,c);fp=fingerprint({'v':1,'messages':messages,'model':gen_model,'temperature':0,'top_p':1,'seed':42})
        ans=cached_call(out/'cache/candidate_answer'/qid/f'{cid}.json',fp,lambda:{'answer':calls.chat(messages,'generation',gen_model)},calls,'candidate_answer')
        if ans['status']!='ok':return None
        jp=judge_prompt(b['question_type'],b['question'],b['gold_answer'],ans['answer'],abstention='abs' in qid);jfp=fingerprint({'v':1,'prompt':jp,'model':judge_model});judged=cached_call(out/'cache/candidate_judge'/qid/f'{cid}.json',jfp,lambda:{'raw':calls.chat([{'role':'user','content':jp}],'judge',judge_model),'correct':None},calls,'candidate_judge')
        if judged['status']=='ok' and judged.get('correct') is None:judged['correct']=judge_result(judged['raw']);atomic_write_json(out/'cache/candidate_judge'/qid/f'{cid}.json',judged)
        lp=None
        if scorer.available:
            lfp=fingerprint({'v':1,'messages':messages,'gold':b['gold_answer'],'model':gen_model});lp=cached_call(out/'cache/candidate_logprob'/qid/f'{cid}.json',lfp,lambda:scorer.score(messages,b['gold_answer']),calls,'candidate_logprob')
        bc=bool(base['judge']['correct']);cc=bool(judged.get('correct'));continuous=bool(lp and lp['status']=='ok' and base['logprob'] and base['logprob']['status']=='ok');da=int(cc)-int(bc);dl=(lp['mean_logprob_per_token']-base['logprob']['mean_logprob_per_token']) if continuous else None
        return {'question_id':qid,'question':b['question'],'question_type':b['question_type'],'base_memory_ids':b['base_memory_ids'],'candidate_memory_id':cid,'candidate_text':c['candidate_text'],'session_id':c.get('session_id'),'timestamp':c.get('timestamp'),'original_hebbian_score':c.get('original_hebbian_score'),'original_spreading_score':c.get('original_spreading_score'),'baseline_answer':base['answer']['answer'],'candidate_answer':ans['answer'],'baseline_correct':bc,'candidate_correct':cc,'transition':transition(bc,cc),'baseline_gold_total_logprob':base['logprob'].get('total_logprob') if continuous else None,'candidate_gold_total_logprob':lp.get('total_logprob') if continuous else None,'baseline_gold_mean_logprob':base['logprob'].get('mean_logprob_per_token') if continuous else None,'candidate_gold_mean_logprob':lp.get('mean_logprob_per_token') if continuous else None,'delta_answer':da,'delta_gold_logprob':(lp['total_logprob']-base['logprob']['total_logprob']) if continuous else None,'delta_gold_mean_logprob':dl,'continuous_signal_available':continuous,'utility_score':dl if continuous else da,'utility_source':'delta_gold_mean_logprob' if continuous else 'delta_answer','semantic_score':c.get('semantic_score'),'hebbian_score':c.get('hebbian_score'),'spreading_score':c.get('spreading_score'),'source_edges':c['source_edges'],'actual_baseline_context':base['context'],'actual_intervention_context':context}
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,a.workers)) as ex:
        fut={ex.submit(intervene,x):(x[0]['question_id'],x[1]['candidate_memory_id']) for x in candidates}
        for f in concurrent.futures.as_completed(fut):
            try:
                row=f.result(); rows.extend([row] if row else [])
            except Exception as e:
                qid,cid=fut[f];failures.append({'question_id':qid,'candidate_memory_id':cid,'stage':'intervention','error':f'{type(e).__name__}: {e}'[:1000]})
            if len(rows)%25==0:print(f"Interventions: {len(rows)}/{len(fut)}",flush=True)
    rows.sort(key=lambda x:(x['question_id'],x['candidate_memory_id']));dump_jsonl(out/'candidate_utility.jsonl',rows);dump_jsonl(out/'failures.jsonl',failures)
    vals=[r['utility_score'] for r in rows];cont=[r['delta_gold_mean_logprob'] for r in rows if r['continuous_signal_available']];sem=[(r['semantic_score'],r['utility_score']) for r in rows if r['semantic_score'] is not None];heb=[(r['hebbian_score'],r['utility_score']) for r in rows if r['hebbian_score'] is not None];spr=[(r['spreading_score'],r['utility_score']) for r in rows if r['spreading_score'] is not None]
    def corr(p):return {'n':len(p),'pearson':pearson([x for x,y in p],[y for x,y in p]),'spearman':pearson(ranks([x for x,y in p]),ranks([y for x,y in p])) if len(p)>1 else None}
    by_transition=defaultdict(list)
    for row in rows:by_transition[row['transition']].append(row['utility_score'])
    stats={'utility':describe(vals),'delta_gold_mean_logprob':describe(cont),'positive':sum(x>.02 for x in vals),'near_zero':sum(abs(x)<=.02 for x in vals),'negative':sum(x<-.02 for x in vals),'category_epsilon':.02,'utility_by_answer_transition':{key:describe(value) for key,value in sorted(by_transition.items())},'correlations':{'semantic':corr(sem),'hebbian':corr(heb),'spreading':corr(spr)}};atomic_write_json(out/'utility_statistics.json',stats)
    sem_quartiles=statistics.quantiles([x for x,y in sem],n=4,method='inclusive') if len(sem)>1 else None;sem_lo=sem_quartiles[0] if sem_quartiles else None;sem_hi=sem_quartiles[2] if sem_quartiles else None;up75=statistics.quantiles(vals,n=4,method='inclusive')[2] if len(vals)>1 else None;pair_stats={}
    for eps,name in zip(EPSILONS,('eps0','eps002','eps005','eps010')):
        ps=build_pairs(rows,eps,sem_lo,sem_hi,up75);dump_jsonl(out/f'pairwise_preferences_{name}.jsonl',ps);pair_stats[str(eps)]={'count':len(ps),'question_count':len({p['question_id'] for p in ps}),'type_counts':dict(Counter(t for p in ps for t in p['pair_types']))}
    atomic_write_json(out/'pair_statistics.json',pair_stats);splits,profiles=stratified_split(rows);[atomic_write_json(out/f'{k}_question_ids.json',v) for k,v in splits.items()]
    split_profile_counts={split:{label:sum(profiles[qid][label] for qid in ids) for label in ('has_positive','has_negative','has_within_question_variation')}|{'candidate_count':sum(profiles[qid]['candidate_count'] for qid in ids)} for split,ids in splits.items()}
    atomic_write_json(out/'split_manifest.json',{'unit':'question_id','population':'candidate-bearing questions only','seed':SPLIT_SEED,'ratios':{'train':.8,'dev':.1,'test':.1},'stratification_threshold':.02,'stratification_labels':['has_positive','has_negative','has_within_question_variation'],'counts':{k:len(v) for k,v in splits.items()},'profile_counts':split_profile_counts,'question_profiles':profiles,'question_ids':splits})
    oracle=json.loads(Path(a.oracle_v05).read_text(encoding='utf-8'));omap={(str(q['question_id']),str(cid)):lab.get('label','uncertain').upper() for q in oracle.get('records',[]) for cid,lab in q.get('oracle_edge_labels',{}).items()};bylab=defaultdict(list)
    for r in rows:
        label=omap.get((r['question_id'],r['candidate_memory_id']))
        if label:bylab[label].append(r['utility_score'])
    def prob(a,b):return sum(x>y for x in a for y in b)/(len(a)*len(b)) if a and b else None
    oracle_analysis={'matched':sum(map(len,bylab.values())),'by_label':{k:describe(v) for k,v in bylab.items()},'p_supporting_gt_redundant':prob(bylab['SUPPORTING'],bylab['REDUNDANT']),'p_supporting_gt_irrelevant':prob(bylab['SUPPORTING'],bylab['IRRELEVANT'])};atomic_write_json(out/'oracle_cross_validation.json',oracle_analysis)
    sizes=[len(b['candidates']) for b in bases];summary={'total_questions':len(bases),'questions_with_candidates':sum(x>0 for x in sizes),'candidate_total':len(candidates),'average_candidates_per_question':statistics.mean(sizes) if sizes else 0,'median_candidates_per_question':statistics.median(sizes) if sizes else 0,'average_candidates_per_candidate_question':len(candidates)/max(1,sum(x>0 for x in sizes)),'median_candidates_per_candidate_question':statistics.median([x for x in sizes if x]) if any(sizes) else 0,'max_candidates':max(sizes,default=0),'transitions':dict(Counter(r['transition'] for r in rows)),'continuous_signal_available':scorer.available,'continuous_signal_reason':scorer.reason,'calls':dict(calls.c),'failures':len(failures)};atomic_write_json(out/'intervention_summary.json',summary)
    manifest={'protocol':'base_conditioned_utility_dataset_v1_1','candidate_definition':'unique one-hop non-Base neighbors of frozen Base Top-15 in the encoded Hebbian graph','utility_formula':'delta_gold_mean_logprob when available, else delta_answer','reader_model':gen_model,'judge_model':judge_model,'temperature':0,'top_p':1,'seed':42,'split_seed':SPLIT_SEED,'split_population':'candidate-bearing questions only','split_counts':{k:len(v) for k,v in splits.items()},'dataset_sha256':sha256_file(a.data_path),'git_commit':git_commit(),'inputs':vars(a),'outputs':[p.name for p in out.iterdir() if p.is_file()]};atomic_write_json(out/'dataset_manifest.json',manifest)
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(f'report\t{out}')
if __name__=='__main__':main()
