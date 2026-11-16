"""Fig. 4: measured random-store accuracy and both registered predictions."""
from plot_common import *

def main():
    all_rows=read('fig_store.csv'); rows=subset(all_rows,policy='random')
    Ts=sorted(set(int(r['T']) for r in rows))
    readers=['qwen2.5-14b','phi-4']
    fig,axes=plt.subplots(1,len(Ts),figsize=(7.16,2.25),sharey=True)
    fig.subplots_adjust(left=.075,right=.985,top=.905,bottom=.28,wspace=.12)
    for ax,T,letter in zip(axes,Ts,'abc'):
        decorate(ax)
        ax.set_ylim(.40,1.)
        ax.set_yticks([.4,.6,.8,1.])
        ax.text(0,1.045,f'({letter}) $T = {T}$',transform=ax.transAxes,
                ha='left',va='bottom',weight='bold',fontsize=8)
        for reader,offset in zip(readers,[-.012,.012]):
            data=sorted(subset(rows,reader=reader,T=T),key=lambda r:float(r['f']))
            x=values(data,'f'); color=COLORS[reader]; marker=MARKERS[reader]
            for col,style,width in [('pred_store','-',2.1),('pred_proj','--',1.6)]:
                ax.plot(x,values(data,col),color=color,linestyle=style,linewidth=width,
                        zorder=4)
            # CI bars retain exact CSV f; only measured glyphs are dodged.
            errorbar(ax,x,values(data,'obs'),values(data,'lo'),values(data,'hi'),
                     color=color,fmt='none',zorder=3,alpha=.85,
                     elinewidth=.95,capsize=2.5,capthick=.95)
            ax.plot(x+offset,values(data,'obs'),color=color,marker=marker,
                    linestyle='none',zorder=5,markersize=MARKER_SIZES[reader],
                    markeredgecolor='white',markeredgewidth=.7,clip_on=False)
        ax.axhline(.5,color='#666666',linestyle=':',linewidth=1.,zorder=1)
        ticks=sorted(set(values(subset(rows,T=T),'f')))[::2]
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: f'{x:.0f}' if x in (0,1) else f'{x:.2f}'))
        ax.set_xlim(min(ticks)-.04,max(ticks)+.04)
    axes[0].set_ylabel('Balanced accuracy',labelpad=3)
    fig.supxlabel('Store capacity / mean state size, $f$',x=.53,y=.1386,fontsize=8)
    from matplotlib.container import ErrorbarContainer
    from matplotlib.collections import LineCollection
    measured=ErrorbarContainer((
        Line2D([],[],color='#333333',marker='o',markersize=3,linestyle='none'),
        (Line2D([],[],color='#333333',marker='_',markersize=5,markeredgewidth=.95),),
        (LineCollection([],colors='#333333',linewidths=.95),)),
        has_yerr=True,label='measured (95% CI)')
    handles=[reader_handle(r) for r in readers]+[
        measured,Line2D([],[],color='.2',linewidth=2.1,label='store prediction (registered)'),
        Line2D([],[],color='.2',linestyle='--',linewidth=1.6,label='projection prediction')]
    fig.legend(handles=handles,ncol=5,loc='lower center',bbox_to_anchor=(.53,.0505),
               columnspacing=1.05,handlelength=1.85)
    export(fig,'fig4_store',{'csv_rows':len(all_rows),'random_rows_used':len(rows),
           'nonrandom_rows_excluded':len(all_rows)-len(rows),'panels':len(Ts),
           'measured_series':len(Ts)*len(readers),'measured_points':len(rows),
           'store_prediction_series':len(Ts)*len(readers),'store_prediction_vertices':len(rows),
           'projection_prediction_series':len(Ts)*len(readers),'projection_prediction_vertices':len(rows),
           'ci_below_axis':sum(float(r['lo'])<float(axes[0].get_ylim()[0]) for r in rows)})

if __name__ == '__main__': main()
