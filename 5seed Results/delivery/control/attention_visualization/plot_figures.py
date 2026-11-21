"""Publication-friendly PNG/SVG figures from the verified exported tensors."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODES=('unshuffled','shuffle_3','shuffle_4','mlp')
LABELS=('Unshuffled','Shuffle first 3','Shuffle all 4','MLP')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
    'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);args=ap.parse_args()
    root=args.directory;out=root/'figures';out.mkdir(exist_ok=True)
    data=np.load(root/'aggregates.npz');figures=[]
    def mean(ds,mode,name):
        return np.mean([data[f'{ds}_{seed}_{mode}_{name}'] for seed in range(42,47)],axis=0)
    def save(fig,name,description):
        fig.savefig(out/f'{name}.png',dpi=230,bbox_inches='tight')
        fig.savefig(out/f'{name}.svg',bbox_inches='tight')
        plt.close(fig);figures.append(dict(name=name,description=description,png=f'figures/{name}.png',svg=f'figures/{name}.svg'))
    for ds in ('beauty','toys'):
        fig,axes=plt.subplots(2,4,figsize=(16.5,7.6),layout='constrained')
        mats=[mean(ds,m,'recency')[-1].mean(0) for m in MODES]
        vmax=max(x.max() for x in mats)
        for j,(mode,label) in enumerate(zip(MODES,LABELS)):
            ax=axes[0,j];im=ax.imshow(mats[j],vmin=0,vmax=vmax,cmap='viridis',aspect='auto',interpolation='nearest')
            ax.set_title(label);ax.set_xticks([0,5,10,15,19],['20','15','10','5','1'])
            ax.set_yticks(range(4),['H0','H1','H2','H3']);ax.set_xlabel('History item lag (1 = most recent)')
            if j==0:ax.set_ylabel('Next SID digit / cross-attention')
            ax=axes[1,j];dec=mean(ds,mode,'decoder_self')[-1].mean(0)
            masked=np.ma.array(dec,mask=np.triu(np.ones((4,4),bool),1))
            im2=ax.imshow(masked,vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
            ax.set_xticks(range(4),['BOS','H0','H1','H2']);ax.set_yticks(range(4),['H0','H1','H2','H3'])
            ax.set_xlabel('Generated-prefix key')
            if j==0:ax.set_ylabel('Next SID digit / decoder self-attention')
        fig.colorbar(im,ax=axes[0,:],label='Attention mass per history item',shrink=.8)
        fig.colorbar(im2,ax=axes[1,:],label='Attention probability',shrink=.8)
        fig.suptitle(f'{ds.title()} | Last layer, mean of 6 heads\n100 fixed users per seed; equal-weight mean of five seeds (42-46)',fontsize=15)
        fig.supxlabel('Cross-attention follows each condition\'s generated Top1 prefix. Missing older positions contribute zero; item mass includes SEP.',fontsize=10)
        save(fig,f'{ds}_five_seed_overview',f'{ds}: last-layer cross-attention by item recency and decoder self-attention, five-seed mean')
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.8),layout='constrained')
    mats={(ds,m):mean(ds,m,'sid_mass')[-1].mean(0) for ds in ('beauty','toys') for m in MODES[:3]}
    vmax=max(x.max() for x in mats.values())
    for i,ds in enumerate(('beauty','toys')):
        for j,m in enumerate(MODES[:3]):
            ax=axes[i,j];a=mats[ds,m];im=ax.imshow(a,vmin=0,vmax=vmax,cmap='viridis',aspect='auto')
            ax.set_xticks(range(5),['H0','H1','H2','H3','SEP']);ax.set_yticks(range(4),['H0','H1','H2','H3'])
            ax.set_title(f'{ds.title()} | {LABELS[j]}')
            ax.set_xlabel('Original SID identity after inverse permutation')
            if j==0:ax.set_ylabel('Next SID digit')
            for row in range(4):
                for col in range(5):ax.text(col,row,f'{a[row,col]:.2f}',ha='center',va='center',color='white' if a[row,col]<vmax*.55 else 'black',fontsize=9)
    fig.colorbar(im,ax=axes,label='Attention mass, summed across history items',shrink=.85)
    fig.suptitle('Does attention follow a SID identity or its physical position?\nLast layer; six-head mean; 100 users and five seed means',fontsize=15)
    fig.supxlabel('Columns are restored to original H0-H3 identity. MLP has no individual SID keys and is therefore excluded.',fontsize=10)
    save(fig,'sid_identity_alignment','Cross-attention mass by original SID identity after inverse permutation')
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),layout='constrained')
    for ax,ds in zip(axes,('beauty','toys')):
        for i,m in enumerate(('shuffle_3','shuffle_4')):
            ys=np.array([data[f'{ds}_{s}_{m}_item_tv'][-1].mean() for s in range(42,47)])
            ax.scatter(i+np.linspace(-.12,.12,5),ys,color=('#3366aa','#c86a37')[i],zorder=3)
            ax.errorbar(i,float(ys.mean()),yerr=float(ys.std(ddof=1)),color='black',fmt='_',capsize=5,markersize=15)
            for offset,y,seed in zip(np.linspace(-.12,.12,5),ys,range(42,47)):
                ax.annotate(str(seed),(i+offset,y),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
        ax.set_xticks([0,1],['Shuffle first 3','Shuffle all 4']);ax.set_xlim(-.5,1.5);ax.set_ylim(bottom=0)
        ax.set_title(ds.title());ax.set_ylabel('Mean item-attention total variation distance');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Change from unshuffled cross-attention | Last layer',fontsize=15)
    fig.supxlabel('Each dot = one seed, 100 paired users and 6 heads. Bar = five-seed mean +/- sample SD. Generated prefixes may differ.',fontsize=9)
    save(fig,'paired_attention_change','Paired per-seed changes from unshuffled item-level cross-attention; descriptive, not causal')
    representatives=torch.load(root/'representatives.pt',map_location='cpu',weights_only=False)
    for ds in ('beauty','toys'):
        for kind in ('decoder_cross','encoder_self'):
            fig,axes=plt.subplots(1,4,figsize=(16,5.4),layout='constrained');mats=[]
            for m in MODES:
                record=representatives[(ds,m)];a=record['arrays'][kind][-1].mean(0);n=record['n'];stride=record['payload']['stride']
                if kind=='decoder_cross':a=a.reshape(4,n,stride).sum(-1)
                else:a=a.reshape(n,stride,n,stride).sum(-1).mean(1)
                mats.append(a)
            vmax=max(a.max() for a in mats)
            for j,(ax,a) in enumerate(zip(axes,mats)):
                im=ax.imshow(a,vmin=0,vmax=vmax,cmap='viridis',interpolation='nearest',aspect='auto')
                ax.set_title(LABELS[j]);n=a.shape[1];ax.set_xticks(range(n),[str(i+1) for i in range(n)])
                ax.set_xlabel('History item (oldest to newest)')
                if kind=='decoder_cross':ax.set_yticks(range(4),['H0','H1','H2','H3'])
                else:ax.set_yticks(range(n),[str(i+1) for i in range(n)])
                if j==0:ax.set_ylabel('Next SID digit' if kind=='decoder_cross' else 'Query history item')
            uid=representatives[(ds,'unshuffled')]['uid']
            fig.colorbar(im,ax=axes,label='Attention mass per item',shrink=.8)
            fig.suptitle(f'{ds.title()} | User {uid}, seed 42 | {kind.replace("_"," ")}\nLast layer; six-head mean; item-level aggregation',fontsize=14)
            fig.supxlabel('User selected by median history length, without looking at predictions. Shuffle item key mass includes its SEP token.',fontsize=9)
            save(fig,f'{ds}_user_{uid}_{kind}',f'{ds}: median-history user {uid}, seed42, {kind} at item level')
    (root/'figure_index.json').write_text(json.dumps(figures,indent=2))
    print(json.dumps(dict(figures=len(figures),formats=['PNG 230dpi','SVG']),indent=2))


if __name__=='__main__':main()
