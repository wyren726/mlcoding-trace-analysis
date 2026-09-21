"""Statistical triage of the frozen diagnosis; never edits the taxonomy."""
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from datetime import datetime, timezone
import csv
import hashlib
import json
import numpy as np
from scipy.stats import beta, binom, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/高质量Trace筛选_20260915/子需求精简诊断_20260921'
OUT = ROOT / 'docs/高质量Trace筛选_20260915/子需求精简统计决策_20260921'
SEED, REPS = 20260921, 5000


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def loadcsv(p):
    with p.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def writecsv(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dump(name, value):
    (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def upper(k, n):
    return float(beta.ppf(.95, k+1, n-k)) if n and k < n else (1. if n else None)


def lower(k, n):
    return float(beta.ppf(.05, k, n-k+1)) if k and n else (0. if n else None)


def by_adjust(values):
    """Benjamini–Yekutieli: FDR correction valid under arbitrary dependence."""
    p = np.asarray(values)
    m = len(p)
    ix = np.argsort(p, kind='stable')
    adjusted = np.minimum.accumulate((p[ix]*m*np.sum(1/np.arange(1, m+1))/np.arange(1,m+1))[::-1])[::-1]
    q = np.empty(m)
    q[ix] = np.minimum(adjusted, 1)
    return q


def pc(x):
    return round(100*x, 4) if x is not None else ''


def table(headers, rows):
    def cell(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return ['| '+' | '.join(headers)+' |', '|'+'|'.join(['---']*len(headers))+'|'] + [
        '| '+' | '.join(cell(x) for x in row)+' |' for row in rows]


def main():
    files = [SOURCE/name for name in ['子需求精简诊断表.csv', '逐案例命中与来源.jsonl', '全部子需求两两关系.csv', 'manifest.json']]
    input_hashes = {str(p): sha(p) for p in files}
    previous = json.loads((SOURCE/'manifest.json').read_text())
    for p in files[:-1]:
        assert sha(p) == previous['outputs'][p.name]
    rows = loadcsv(files[0])
    all_cases = [json.loads(line) for line in files[1].open() if line.strip()]
    cases = [c for c in all_cases if c['counted_in_primary']]
    sids = [r['子需求ID'] for r in rows]
    ix = {s:i for i,s in enumerate(sids)}
    assert len(sids) == len(set(sids)) == 216
    assert all(len(c['session_ids']) == 1 for c in cases)
    sessions = sorted({c['session_ids'][0] for c in cases})
    sx = {s:i for i,s in enumerate(sessions)}
    n, g = len(cases), len(sessions)
    X = np.zeros((n,216), dtype=np.int64)
    cluster_sums = np.zeros((g,216), dtype=np.int64)
    cluster_sizes = np.zeros(g, dtype=np.int64)
    for j,c in enumerate(cases):
        k = sx[c['session_ids'][0]]
        for sid in c['subrequirements']:
            X[j,ix[sid]] = 1
        cluster_sums[k] += X[j]
        cluster_sizes[k] += 1
    Y = (cluster_sums > 0).astype(np.int64)
    counts, scounts = X.sum(0), Y.sum(0)
    assert np.array_equal(counts, [int(r['明确命中案例数']) for r in rows])
    # Cluster bootstrap preserves within-session dependence and estimates the episode-weighted rate.
    rng = np.random.default_rng(SEED)
    boot = np.empty((REPS,216))
    for start in range(0,REPS,250):
        weights = rng.multinomial(g, np.full(g,1/g), size=min(250,REPS-start)).astype(float)
        boot[start:start+len(weights)] = (weights @ cluster_sums)/(weights @ cluster_sizes)[:,None]
    lo,hi = np.quantile(boot,[.025,.975],axis=0)
    # Formal rarity tests use one Bernoulli observation per session, a distinct estimand.
    p_rare = binom.cdf(scounts,g,.01)
    q_rare = by_adjust(p_rare)
    duplicates = defaultdict(list)
    for c in all_cases:
        duplicates[c['representative_case_id']].append(set(c['subrequirements']))
    bounds = {sid:(sum(all(sid in s for s in ss) for ss in duplicates.values()),
                    sum(any(sid in s for s in ss) for ss in duplicates.values())) for sid in sids}

    prior_pairs = loadcsv(files[2])
    co = Y.T @ Y
    pair_rows, p_assoc = [], []
    for old,(a,b) in zip(prior_pairs,combinations(range(216),2)):
        assert old['A'] == sids[a] and old['B'] == sids[b]
        common = int(co[a,b]); na,nb = int(scounts[a]),int(scounts[b])
        p = float(fisher_exact([[common,na-common],[nb-common,g-na-nb+common]], alternative='greater').pvalue)
        p_assoc.append(p)
        ea,eb,ec = int(counts[a]),int(counts[b]),int(old['共同命中数'])
        same = float(old['同需求共现占比'] or 0)/100
        sa,sb = common/na if na else 0, common/nb if nb else 0
        ab,ba = ec/ea if ea else 0, ec/eb if eb else 0
        signal,direction = '无强调整信号',''
        merge = ec>=20 and common>=20 and ab>=.8 and ba>=.8 and sa>=.8 and sb>=.8 and same>=.8
        if merge:
            signal = '合并初筛，语义未核查'
        # Require consistent direction at both episode and session levels.
        for source,target,forward,reverse,sforward,sreverse,total in [
            (a,b,ab,ba,sa,sb,na),(b,a,ba,ab,sb,sa,nb)]:
            if ec>=20 and common>=20 and forward>=.9 and reverse<=.5 and sforward>=.9 and sreverse<=.5:
                direction = f'{sids[source]} → {sids[target]}'
                signal = '归属初筛，语义未核查' if same>=.8 else '流程共现，不能据此归入'
        pair_rows.append(dict(**old, 会话共同数=common, 会话A总数=na, 会话B总数=nb,
            会话A到B单侧95下界百分比=pc(lower(common,na)), 会话B到A单侧95下界百分比=pc(lower(common,nb)),
            会话正关联Fisher_p=p, 会话正关联BY_q=0.,
            片段A到B单侧95下界参考百分比=pc(lower(ec,ea)), 片段B到A单侧95下界参考百分比=pc(lower(ec,eb)),
            升级信号=signal, 升级方向=direction, 强合并统计门槛=False, 强归入统计门槛=False))
    q_assoc = by_adjust(p_assoc)
    for r,q in zip(pair_rows,q_assoc):
        r['会话正关联BY_q'] = float(q)
        r['强合并统计门槛'] = bool(r['升级信号'].startswith('合并') and q<=.05
            and min(r['会话A到B单侧95下界百分比'],r['会话B到A单侧95下界百分比'])>=80)
        if r['升级信号'].startswith('归属'):
            ci = r['会话A到B单侧95下界百分比'] if r['升级方向'].startswith(r['A']) else r['会话B到A单侧95下界百分比']
            r['强归入统计门槛'] = bool(q<=.05 and ci>=80)
    result = []
    for j,r in enumerate(rows):
        sid,pid = r['子需求ID'],r['父需求ID']
        siblings = [k for k,x in enumerate(rows) if x['父需求ID']==pid]
        parent_sessions = int(np.any(Y[:,siblings],axis=1).sum())
        parent_upper = upper(int(scounts[j]),parent_sessions)
        episode_upper = upper(int(counts[j]),n)
        session_upper = upper(int(scounts[j]),g)
        protected = pid=='req_036' or any(w in r['子需求'] for w in ['权限','合规','脱敏','污染','回滚','伦理'])
        stable_rare = bool(q_rare[j]<=.05 and session_upper<.01 and episode_upper<.01)
        category,reason,gap = '保留','当前未达到低频或关系调整条件','未证明可以无损精简'
        candidates = [x for x in pair_rows if sid in [x['A'],x['B']] and x['升级信号']!='无强调整信号']
        if counts[j]/n<.01:
            category='证据不足'
            reason='低频，但低频本身不足以决定删除或归入'
            gap='需查漏标、独立验收价值与可承接类别'
            if stable_rare and counts[j]==0 and parent_upper is not None and parent_upper<.05 and not protected and int(r['仅未明确支持案例数'])==0:
                category='删除候选'
                reason='零命中；会话低频检验通过BY；父内会话率上界<5%'
                gap='仅通过频率门槛；尚需相关未命中原文抽查及独立价值核查，不可直接删除'
            elif protected:
                category='保留'
                reason='具有风险/合规相关语义，低频不作为删除依据（暂定保护规则）'
                gap='如需调整，先确认专项覆盖是否仍可表达'
            elif not stable_rare:
                reason='未同时通过会话低频BY检验与两种出现率上界条件'
            elif parent_upper is None or parent_upper>=.05:
                reason='总体低频，但父需求内仍缺少稳定低频证据'
        if any(x['强合并统计门槛'] for x in candidates):
            category='合并成新项候选';reason='通过双向重叠统计门槛';gap='需原文证明语义等价，现有名称能否复用尚待判断'
        if any(x['强归入统计门槛'] and x['升级方向'].startswith(sid) for x in candidates):
            category='归入已有项候选';reason='通过单向包含统计门槛';gap='需证明语义包含且不损失独立验收条件'
        result.append(dict(**r,
            片段率单侧95精确上界参考百分比=pc(episode_upper),
            会话命中数=int(scounts[j]), 会话总数=g, 会话率单侧95精确上界百分比=pc(session_upper),
            会话低频检验_p=float(p_rare[j]), 会话低频检验_BY_q=float(q_rare[j]), 通过严格低频门槛=stable_rare,
            会话聚类bootstrap片段率95下界百分比=pc(float(lo[j])) if scounts[j]>=10 else '',
            会话聚类bootstrap片段率95上界百分比=pc(float(hi[j])) if scounts[j]>=10 else '',
            bootstrap适用说明='命中会话不足10个，不报告退化或不稳定的bootstrap区间' if scounts[j]<10 else '双侧95% percentile；5000次按会话成组重采样',
            父需求命中会话数=parent_sessions, 父内会话率单侧95上界百分比=pc(parent_upper),
            重复版本最少可能命中数=bounds[sid][0], 重复版本最多可能命中数=bounds[sid][1],
            暂定保护规则命中=protected, 统计决策类别=category, 决策依据=reason, 尚缺证据=gap,
            关系统计结论='；'.join(x['升级方向']+'：'+x['升级信号'] for x in candidates),
            语义审核状态='未开展独立原文复核', 是否可执行删改=False))
    summary = dict(episodes=n,sessions=g,children=len(rows),bootstrap_repetitions=REPS,seed=SEED,
        rarity_fdr_method='Benjamini–Yekutieli',rarity_family=216,association_family=len(pair_rows),
        strict_rare=sum(r['通过严格低频门槛'] for r in result),
        categories=dict(Counter(r['统计决策类别'] for r in result)),
        positive_association_by_significant=sum(q_assoc<=.05).item(),
        strong_merge_pairs=sum(r['强合并统计门槛'] for r in pair_rows),
        strong_absorption_pairs=sum(r['强归入统计门槛'] for r in pair_rows),
        executable_changes=0)
    assert [r['子需求ID'] for r in result]==sids
    assert len(pair_rows)==23220 and np.all(q_rare>=p_rare-1e-12)
    assert np.all(q_assoc>=np.asarray(p_assoc)-1e-12)
    assert np.all(counts>=scounts) and np.all((lo>=0)&(hi<=1)&(lo<=hi))
    OUT.mkdir(parents=True,exist_ok=False)
    writecsv('子需求统计决策表.csv',result)
    writecsv('全部两两关系统计检验.csv',pair_rows)
    signal_rows=[r for r in pair_rows if r['升级信号']!='无强调整信号']
    if signal_rows:writecsv('关系调整核查表.csv',signal_rows)
    dump('统计摘要.json',summary)
    # A compact queue of already labeled evidence; deletion requires additional negative-case search.
    queue=[]
    for r in result:
        if r['统计决策类别']=='保留':continue
        sid=r['子需求ID'];positive=[c for c in cases if sid in c['subrequirements']][:3]
        siblings={x['子需求ID'] for x in rows if x['父需求ID']==r['父需求ID']}
        negative=[c for c in cases if sid not in c['subrequirements'] and siblings.intersection(c['subrequirements'])][:3]
        queue.append(dict(subrequirement_id=sid,name=r['子需求'],category=r['统计决策类别'],
            definition=r['定义'],boundary=r['边界'],gap=r['尚缺证据'],
            positive_examples=[dict(case_id=c['case_id'],goal=c['goal'],label_path=c['label_path'],
                label_line=c['label_line'],mappings=c['mappings'][sid]) for c in positive],
            parent_negative_examples=[dict(case_id=c['case_id'],goal=c['goal'],label_path=c['label_path'],
                label_line=c['label_line']) for c in negative],
            sampling_notice='以上是固定顺序定位样例，不是随机审计样本，不能用于估计漏标率'))
    dump('语义核查队列.json',queue)
    md=['# 子需求精简：统计决策版','',
        f'覆盖{n}个去重任务片段、{g}个会话、216项子需求。完整表严格保持原顺序。所有结论是探索性核查建议，本次没有更改分类体系。','',
        '## 结果','']
    md+=table(['类别','项数'],summary['categories'].items())
    md += ['',f'严格低频门槛通过{summary["strict_rare"]}项；正关联经BY校正显著的关系{summary["positive_association_by_significant"]}对，但强合并候选{summary["strong_merge_pairs"]}对、强归入候选{summary["strong_absorption_pairs"]}对。关联显著不等于可合并。', '',
        '“重组”需要独立标注的混淆/争议数据，现有命中记录不能估计标注一致性，因此未自动产生重组结论。删除候选也只通过统计门槛，不代表语义价值已经审核。','',
        '## 指标与规则','',
        '- 片段出现率=命中片段数/1129；会话出现率=至少命中一次的会话数/940。两者是不同统计目标，不互相替代。',
        '- 片段单侧95% Clopper–Pearson上界仅作独立二项假设下的参考。正式低频检验以每会话一个0/1值：H0为会话出现率≥1%，H1为<1%，p=P[Binomial(940,0.01)≤观测命中数]。216项一起BY校正。',
        '- 严格低频门槛：会话BY q≤0.05、会话率上界<1%、片段率参考上界<1%。上界本身是逐项95%，不是多重比较校正后的同时区间；q负责检验家族错误控制。',
        '- 按会话有放回抽取940个聚类，保留每个会话所有片段，5000次重采样计算片段率双侧95% percentile区间。命中会话不足10个时不展示bootstrap区间，特别是零命中不能报告[0,0]作为确定性证据。随机种子固定。',
        '- 删除候选进一步要求：零命中、父需求内会话率单侧95%上界<5%、没有仅未明确支持的指向记录、未触发暂定风险保护规则。父内5%是新增的局部频率诊断阈值，不是删除定律，未单独进行父内多重检验。',
        '- 暂定保护规则：req_036及名称含权限、合规、脱敏、污染、回滚、伦理的项，暂保留；这是业务价值启发式，不是统计发现，也不表示其他低频项无价值。',
        '- 所有23220对关系在会话二元命中表上做单侧Fisher正关联检验，并统一BY校正；零覆盖项对应检验保留在家族中。BY允许标签之间任意依赖，较BH保守，但仍依赖会话作为独立观测的假设。',
        '- 合并初筛：至少20个共同片段且20个共同会话，片段和会话两个方向比例均≥80%，同一具体需求共现比例≥80%。强统计候选还要求会话两个方向单侧95%下界≥80%及关联q≤0.05。',
        '- 归入初筛：至少20个共同片段且20个共同会话，片段与会话都满足同方向≥90%、反向≤50%，同需求共现≥80%。强候选还要求正向会话95%下界≥80%及关联q≤0.05。上述区间未经全对同时校正，关系候选需要独立数据确认。',
        '- 重复版本给出最少/最多命中敏感性范围，主统计仍使用上一版固定来源选择；不合并不同版本标签制造共现。',
        '- 双向重叠不决定最终名称：语义等价且已有名称准确时可复用名称；已有定义不能涵盖时才拟定新项。单向统计包含也不能替代语义包含。', '',
        '## 数据适用边界','',
        '这是前序筛选后的已打标样本，不是随机抽取的全部用户需求。123条打标失败和58条边界异常没有被当作未命中。同一用户的不同会话仍可能相关，目前没有用户级聚类；BY不能消除采样偏差、跨会话依赖或系统性漏标。批次不平衡通过原表的分批命中率展示，bootstrap未重加权为等批样本。', '',
        '当前所有数据都已用于探索，不能再把其中一部分声称为未看过的验证集。阈值是初始规则；确定删改应使用新样本或独立原文审计确认。语义核查队列只提供定位样例，没有把已有模型证据冒充人工审核。', '',
        '## 完整决策表（原分类顺序）','']
    md+=table(['ID','子需求','片段命中','会话命中','会话率上界%','低频BY q','父内上界%','类别','依据'],
        [[r['子需求ID'],r['子需求'],r['明确命中案例数'],r['会话命中数'],r['会话率单侧95精确上界百分比'],
          f'{r["会话低频检验_BY_q"]:.4g}',r['父内会话率单侧95上界百分比'],r['统计决策类别'],r['决策依据']] for r in result])
    md+=['','## 关系核查','']
    md+=table(['A','B','方向','会话正关联BY q','结论'],[[r['A名称'],r['B名称'],r['升级方向'],f'{r["会话正关联BY_q"]:.4g}',r['升级信号']] for r in signal_rows])
    md+=['','## 下一步语义核查标准','',
        '删除：检查父需求相关的未命中原文是否漏标，并判断独立交付物、验收条件和重要风险是否丢失。低频但有价值时优先收拢为属性，不自动丢弃。', '',
        '合并/归入：抽查共同命中及各自独立案例，判断动作、对象、产物与验收要求是否等价或包含。同一case_requirement_id可能仍是复合需求，不能自动当成同义证据。', '',
        '重组：需要对新旧体系做独立重复标注，比较争议率、无法归类率与关键覆盖；现有记录不足以计算这些指标。建议容忍界限仍沿用前述方案：兜底率增幅不超过2个百分点、争议率相对下降至少20%，同时报告区间与样本量。', '',
        '## 附件','', '- [完整统计决策CSV](子需求统计决策表.csv)', '- [全部23220对关系检验](全部两两关系统计检验.csv)',
        '- [关系调整核查表](关系调整核查表.csv)', '- [语义核查队列和定位样例](语义核查队列.json)',
        '- [统计摘要](统计摘要.json)', '- [版本和文件校验](manifest.json)', '']
    (OUT/'子需求精简统计决策报告.md').write_text('\n'.join(md))
    assert all(sha(Path(p))==h for p,h in input_hashes.items())
    dump('manifest.json',dict(generated_at=datetime.now(timezone.utc).isoformat(),inputs=input_hashes,
        generator=str(Path(__file__).resolve()),generator_sha256=sha(Path(__file__)),summary=summary,
        numpy_version=np.__version__,validation=dict(order_preserved=True,all_pair_count=23220,
            counts_match_previous=True,session_partition_verified=True,by_q_not_below_p=True),
        outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    main()
