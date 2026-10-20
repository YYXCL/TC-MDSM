'Summarize completion rates and successful-request latency in four experiment figures.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from model.types import METHODS

METHOD_LABELS = {"HM-MATAS": "TARGETNET"}

def report(out, paper_style=False):
    out=Path(out);parameters=json.loads((out/'parameters.json').read_text(encoding='utf-8'))
    records=[];decisions=[]
    for file in sorted((out/'instances').rglob('*.json')):
        d=json.loads(file.read_text(encoding='utf-8'));records.extend(d['rows'])
        r=d['rows'][0];p=d['plans'][0]
        decisions.append({k:r[k] for k in ['seed','method','block','x','instance']}|{'decision_s':p['elapsed'],'status':p['status'],'accepted':sum(p['admitted'].values()),'requests':len(d['rows'])}|p.get('timings',{}))
    if not records:raise ValueError('No evaluation results are available for plotting')
    folder=out/'figures';folder.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(records);batch=pd.DataFrame(decisions)

    df.to_csv(folder/'requests.csv',index=False,encoding='utf-8-sig');batch.to_csv(folder/'batch_decisions.csv',index=False,encoding='utf-8-sig')

    decision=batch.groupby(['block','x','method']).decision_s.agg(mean='mean',p50='median',p95=lambda x:x.quantile(.95),batches='count').reset_index()
    decision.to_csv(folder/'decision_latency.csv',index=False,encoding='utf-8-sig')
    def ci(values):

        a=np.array(values,dtype=float);a=a[np.isfinite(a)]
        if not len(a):return np.nan,np.nan
        return a.mean(),t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)) if len(a)>1 else np.nan
    stats=[]
    for (block,x,method),g in df.groupby(['block','x','method']):

        first=g.groupby(['seed','instance']).on_time.mean().groupby('seed').mean()

        latency=g[g.on_time==1].groupby('seed').completion_s.mean()
        rate,width=ci(first);seconds,error=ci(latency)
        stats.append(dict(block=block,x=x,method=method,completion_rate=rate,completion_rate_ci=width,
                          admitted_rate=g.groupby('seed').initial_admitted.mean().mean(),
                          executed_completion_rate=g.groupby('seed').completed.mean().mean(),
                          latency_s=seconds,latency_ci=error,latency_seeds=len(latency),successes=int(g.on_time.sum()),
                          requests=len(g),penalized_s=g.groupby('seed').penalized_s.mean().mean()))
    summary=pd.DataFrame(stats);summary.to_csv(folder/'summary.csv',index=False,encoding='utf-8-sig')

    plt.rcParams.update({'font.sans-serif':['DejaVu Sans'],'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.prop_cycle':plt.cycler(color=['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00','#56B4E9'])})
    if paper_style:
        plt.rcParams.update({'font.family':'DejaVu Sans','font.sans-serif':['DejaVu Sans']})
    methods=parameters['methods']
    markers=['o','s','^','D','v','P']

    metrics=[('completion_rate','completion_rate_ci','On-time completion rate'),
             ('latency_s','latency_ci','Average completion latency (s)')]
    if paper_style:
        metrics=[('completion_rate','completion_rate_ci','On-time completion rate'),
                 ('latency_s','latency_ci','Average completion latency (s)')]
    def save(fig,name,top=1.):

        fig.tight_layout(rect=(0,0,1,top))
        for ext in ('png','pdf'):
            fig.savefig(folder/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)

    complexity=summary[summary.block=='complexity']
    xs=sorted(complexity.x.unique())

    for number,(metric,error,ylabel) in enumerate(metrics,1):
        fig,ax=plt.subplots(figsize=(8.5,4.8))
        width=.13
        for j,method in enumerate(methods):
            g=complexity[complexity.method==method].set_index('x').reindex(xs)
            y=g[metric].to_numpy(dtype=float)
            e=np.nan_to_num(g[error].to_numpy(dtype=float),nan=0.)

            e=np.vstack([np.minimum(e,np.nan_to_num(y)),
                         np.minimum(e,np.nan_to_num(1-y)) if metric=='completion_rate' else e])
            errors={} if paper_style else dict(yerr=e,capsize=2,error_kw={'elinewidth':1.1,'capthick':1.1})
            ax.bar(np.arange(len(xs))+(j-(len(methods)-1)/2)*width,y,width,**errors,
                   edgecolor='white',linewidth=.6,label=METHOD_LABELS.get(method,method))
        ax.set_xticks(np.arange(len(xs)),(['Easy','Medium','Hard'] if paper_style else ['Easy','Medium','Hard']))
        ax.set_xlabel('Task complexity' if paper_style else 'Task complexity');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(ncol=3,fontsize=9,loc='upper center',bbox_to_anchor=(.5,1.20),frameon=False)
        if metric=='completion_rate':ax.set_ylim(0,1.02)
        else:ax.set_ylim(bottom=0)
        save(fig,f'{number}.{ "completion_rate" if number==1 else "completion_latency" }')

    for number,block,xlabel,name in ((3,'concurrency','Concurrent workflows','concurrency'),
                                      (4,'scale','Number of agents','system_scale')):
        if paper_style:
            xlabel='Concurrent workflows' if block=='concurrency' else 'Number of agents'
        sub=summary[summary.block==block]
        xs=sorted(sub.x.unique())
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,(metric,error,ylabel) in zip(axes,metrics):
            for j,method in enumerate(methods):
                g=sub[sub.method==method].set_index('x').reindex(xs)
                y=g[metric].to_numpy(dtype=float)
                e=np.nan_to_num(g[error].to_numpy(dtype=float),nan=0.)
                e=np.vstack([np.minimum(e,np.nan_to_num(y)),
                             np.minimum(e,np.nan_to_num(1-y)) if metric=='completion_rate' else e])
                line=dict(marker=markers[j],markersize=5,linewidth=2 if method=='TC-MDSM' else 1.3,label=METHOD_LABELS.get(method,method))
                if paper_style:ax.plot(xs,y,**line)
                else:ax.errorbar(xs,y,yerr=e,capsize=2,elinewidth=1,**line)
            ax.set(xlabel=xlabel,ylabel=ylabel)
            ax.set_xscale('log',base=2 if paper_style else 10);ax.set_xticks(xs)
            ax.set_xticklabels([str(x) for x in xs]);ax.xaxis.set_minor_formatter(NullFormatter());ax.grid(alpha=.2)
            if paper_style:
                ax.minorticks_off();ax.set_axisbelow(True)
                ax.grid(axis='x',visible=False)
            if metric=='completion_rate':ax.set_ylim(0,1.02)
            else:ax.set_ylim(bottom=0)
            if block=='scale' and metric=='latency_s' and not paper_style:

                ax.set_yscale('symlog',linthresh=20)
                upper=ax.get_ylim()[1]
                ticks=[value for value in [0,20,50,100,200,500,1000] if value<=upper]
                ax.set_yticks(ticks)
                ax.set_yticklabels([str(value) for value in ticks])
                ax.set_ylabel('Average completion latency (s, nonlinear scale)')
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,fontsize=9,frameon=False)
        save(fig,f'{number}.{name}',top=.81)

    display_note = (
        'Figures show means without error bars. CSV tables retain 95% Student-t intervals across seed means. '
        'Line plots use a base-2 logarithmic horizontal axis and linear vertical axes.\n\n'
        if paper_style else
        'Error bars show 95% Student-t intervals across seed means. Displayed completion-rate intervals are clipped to [0, 1]; '
        'latency intervals are clipped at zero. CSV tables retain the original intervals. '
        'The system-scale latency panel uses a symlog vertical axis, linear up to 20 seconds.\n\n'
    )
    (folder/'README.md').write_text(
        '# Experiment results\n\n'
        f"Seeds: {parameters['seeds']}; {parameters['instances']} instances per condition and seed; "
        f"{parameters['counts'][0]} training samples; up to {parameters['epochs']} epochs; "
        f"planning budget: {parameters['seconds']} seconds.\n\n"
        'Completion rate measures first-attempt on-time workflow success. Rejection, execution failure, and lateness count as failures. '
        'Latency includes only on-time successes. Missing latency is not zero latency. Intervals require at least two valid seeds. '
        'Small seed and instance counts can produce substantial variability.\n\n'
        + display_note +
        'Figures summarize saved dynamic-simulation outcomes. Run settings are recorded in parameters.json.\n',
        encoding='utf-8')
    print(str(folder),flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Run directory containing parameters.json and instance results')
    parser.add_argument('--paper-style', action='store_true', help='Plot means without confidence bars')
    args = parser.parse_args()
    report(args.output, paper_style=args.paper_style)
