"""Fig. 3: retention-rule error; exact coincident pairs are merged."""
from plot_common import *

def main():
    rows = read('fig_chain.csv')
    fig, ax = plt.subplots(figsize=(3.5,2.3))
    fig.subplots_adjust(left=.14,right=.975,top=1-4/(2.3*72),bottom=.33)
    decorate(ax)
    groups=[('BM25','salience','#332288','p','-'),
            ('key mention',None,'#6699CC','o','-.'),
            ('recency',None,'#882255','D','--'),
            ('key recency','last applied','#994F00','P',':')]
    handles=[]
    for rule,paired,color,marker,style in groups:
        data=sorted(subset(rows,rule=rule),key=lambda r:int(r['m']))
        if paired:
            other=sorted(subset(rows,rule=paired),key=lambda r:int(r['m']))
            assert np.array_equal(values(data,'m'),values(other,'m'))
            assert np.array_equal(values(data,'error'),values(other,'error'))
        label=rule+', '+paired if paired else rule
        line,=ax.plot(values(data,'m'),values(data,'error'),color=color,marker=marker,
                      linestyle=style,label=label,linewidth=1.8,
                      markersize=6.5 if paired=='last applied' else (5.2 if rule=='recency' else 5.5),
                      markerfacecolor='white' if rule=='key mention' else color,
                      markeredgecolor=color if rule=='key mention' else 'white',
                      markeredgewidth=1.3 if rule=='key mention' else .7,zorder=4)
        handles.append(line)
    ticks=sorted(set(values(rows,'m')))
    # Formula and constants are specified in the brief; no fitted/synthetic data.
    m=np.geomspace(min(ticks),max(ticks),400)
    bound,=ax.plot(m,np.maximum(0,.5-2/(m-1)),color='#222222',linestyle='--',
                   linewidth=2.0,label='Analytic lower bound',zorder=2)
    log_ticks(ax,ticks); ax.set_xlim(4.2,75)
    ax.set_ylim(-.035,.54); ax.set_yticks(np.arange(0,.51,.1))
    ax.set_ylabel('Error',labelpad=3)
    ax.set_xlabel('Events on the queried key, $m$',labelpad=2)
    ax.axhline(.5,color='#888888',linestyle=':',linewidth=.8,zorder=1)
    ax.annotate('Analytic lower bound',xy=(17,.5-2/(17-1)),xytext=(17,.25),
                ha='center',va='center',fontsize=8,fontstyle='italic',color='#222222',
                arrowprops=dict(arrowstyle='-',color='#222222',linewidth=.6,
                                shrinkA=3,shrinkB=2))
    # Two flush-left titled columns form a compact 2-by-2 rule grid.
    for selected,title,x in [(handles[:2],'Order-oblivious',.10),
                             (handles[2:],'Time-aware',.49)]:
        legend=fig.legend(handles=selected,title=title,loc='lower left',
                          bbox_to_anchor=(x,.012),fontsize=7.5,title_fontsize=7.5,
                          labelspacing=.25,handlelength=1.8,alignment='left')
        legend.get_title().set_fontweight('bold')
    export(fig,'fig3_chain',{'csv_rows':len(rows),
           'empirical_csv_rows':len([r for r in rows if r['rule']!='lower bound']),
           'original_rules':len({r['rule'] for r in rows if r['rule']!='lower bound'}),
           'plotted_empirical_series':len(groups),'distinct_plotted_empirical_points':len(groups)*len(ticks),
           'analytic_bound_curves':1,'analytic_render_vertices':len(m)})

if __name__ == '__main__': main()
