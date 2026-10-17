import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R=str(Path(__file__).resolve().parents[1]/'resultados')+'/'
BG='#212121'; FG='#FFFFFF'; MUT='#9E9E9E'; GRID='#424242'; ACC='#64B5F6'; ACC2='#FFB74D'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'text.color':FG,
 'axes.labelcolor':FG,'xtick.color':FG,'ytick.color':FG,'axes.edgecolor':MUT,'font.family':'serif',
 'font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':13,'axes.grid':True,'grid.color':GRID,
 'grid.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
res=pd.read_csv(R+'treino_validacao_todos_casos.csv')
# classes
d=pd.read_csv(R+'distribuicao_classes.csv',index_col=0).iloc[:,0].sort_values()
fig,ax=plt.subplots(figsize=(5.2,3.6)); ax.barh(d.index,d.values,color=ACC,height=.6); ax.grid(axis='y',visible=False)
for i,v in enumerate(d.values): ax.text(v+8,i,f'{v}',va='center',fontsize=12)
ax.set_xlim(0,560); ax.set_xlabel('notícias (após remover duplicatas)'); fig.tight_layout(); fig.savefig(str(Path(__file__).resolve().parent)+'/classes.png',dpi=200)
# alpha
a=pd.read_csv(R+'busca_alpha.csv'); fig,ax=plt.subplots(figsize=(6.2,3.8))
for (f,g),c,mk in zip(a.groupby('familia',sort=False),[ACC,ACC2,FG],['o','s','^']):
    ax.plot(g.alpha,g.validacao_f1_macro,marker=mk,color=c,label=f.replace('LR ',''))
ax.set(xscale='log',xlabel='alpha',ylabel='F1 macro (validação)',ylim=(0,1.02)); ax.legend(loc='lower left'); fig.tight_layout(); fig.savefig(str(Path(__file__).resolve().parent)+'/alpha.png',dpi=200)
# lr curves 2x2
fams=['LR sem regularizacao','LR Ridge','LR Lasso','LR Elastic Net']; names=['Sem regularização','Ridge (α=1e-4)','Lasso (α=1e-5)','Elastic Net (α=1e-5)']
fig,axs=plt.subplots(2,2,figsize=(8.6,4.6),sharex=True,sharey=True)
for ax,f,n in zip(axs.ravel(),fams,names):
    g=res[(res.experimento=='curva_lr')&(res.familia==f)].groupby('n_treino')[['treino_f1_macro','validacao_f1_macro']].agg(['mean','std'])
    for col,c,lab,ls in [('treino_f1_macro',MUT,'Treino','--'),('validacao_f1_macro',ACC,'Validação','-')]:
        m,s=g[col]['mean'],g[col]['std']; ax.plot(g.index,m,ls,marker='o',ms=4,color=c,label=lab); ax.fill_between(g.index,m-s,m+s,color=c,alpha=.2)
    ax.set_title(n,fontsize=13); ax.set_ylim(0.8,1.01)
axs[0,0].legend(loc='lower right',fontsize=11)
for ax in axs[1]: ax.set_xlabel('notícias de treino')
for ax in axs[:,0]: ax.set_ylabel('F1 macro')
fig.tight_layout(); fig.savefig(str(Path(__file__).resolve().parent)+'/curvas_lr.png',dpi=200)
# ad
t=res[res.experimento=='curva_ad'].sort_values('max_depth'); fig,ax=plt.subplots(figsize=(6.2,3.8))
ax.plot(t.max_depth,t.treino_f1_macro,'--o',color=MUT,ms=4,label='Treino'); ax.plot(t.max_depth,t.validacao_f1_macro,'-o',color=ACC2,ms=4,label='Validação')
b=t.loc[t.validacao_f1_macro.idxmax()]; ax.annotate(f'melhor: prof. {int(b.max_depth)}\nF1 = {b.validacao_f1_macro:.2f}',(b.max_depth,b.validacao_f1_macro),xytext=(6.2,0.42),arrowprops=dict(arrowstyle='->',color=FG),fontsize=12)
ax.set(xlabel='profundidade máxima',ylabel='F1 macro',ylim=(0,1.02),xticks=range(1,13)); ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(str(Path(__file__).resolve().parent)+'/curva_ad.png',dpi=200)
# consistencia
c=res[((res.experimento=='curva_lr')&(res.familia=='LR Ridge')&res.fracao_treino.isin([.3,.6,.9]))|(res.experimento=='consistencia_ad')]
g=c.groupby(['familia','fracao_treino']).validacao_f1_macro.agg(['mean','std']).reset_index()
fig,ax=plt.subplots(figsize=(6.2,3.8)); x=np.arange(3); w=.36
for i,(f,col,lab) in enumerate([('LR Ridge',ACC,'LR-GD Ridge'),('AD propria',ACC2,'AD própria (prof. 10)')]):
    s=g[g.familia==f]; ax.bar(x+(i-.5)*w,s['mean'],w,yerr=s['std'],color=col,label=lab,capsize=4,ecolor=FG)
    for xi,m in zip(x,s['mean']): ax.text(xi+(i-.5)*w,m+.025,f'{m:.2f}',ha='center',fontsize=11)
ax.set_xticks(x,['30%\n(382)','60%\n(765)','90%\n(1.148)']); ax.set(ylabel='F1 macro (validação)',ylim=(0,1.22)); ax.grid(axis='x',visible=False); ax.legend(loc='upper left',ncol=2,fontsize=11)
fig.tight_layout(); fig.savefig(str(Path(__file__).resolve().parent)+'/consistencia.png',dpi=200)
# confusion
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('d',[BG,'#1E4E79',ACC])
titles=['LR sem regularização','LR Elastic Net','AD própria (prof. 10)']
fig,axs=plt.subplots(1,3,figsize=(12,4.6))
for i,(ax,tt) in enumerate(zip(axs,titles),1):
    m=pd.read_csv(R+f'confusao_{i}.csv',index_col=0); v=m.values
    ax.imshow(v,cmap=cm,vmin=0,vmax=101); ax.grid(False)
    lab=['bus','ent','pol','spo','tec']
    ax.set_xticks(range(5),lab); ax.set_yticks(range(5),lab if i==1 else ['']*5)
    for a in range(5):
        for b_ in range(5): ax.text(b_,a,v[a,b_],ha='center',va='center',fontsize=12,color=FG if v[a,b_]<60 else '#000000' if False else FG)
    acc=np.trace(v)/v.sum(); ax.set_title(tt+'\nacurácia '+f'{acc:.1%}'.replace('.',','),fontsize=13)
    ax.set_xlabel('predita'); 
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel('real'); [a.tick_params(length=0) for a in axs]; fig.subplots_adjust(left=.05,right=.99,top=.86,bottom=.12,wspace=.05); fig.savefig(str(Path(__file__).resolve().parent)+'/confusao.png',dpi=200)
print('ok')
