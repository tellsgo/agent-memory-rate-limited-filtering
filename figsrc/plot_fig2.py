"""Fig. 2: decay with event count and reader-dependent state advantage."""
from plot_common import *

def main():
    decay, horizon = read('fig_decay.csv'), read('fig_horizon.csv')
    fig, (a,b) = plt.subplots(1, 2, figsize=(7.16, 2.35),
                             gridspec_kw={'width_ratios': [55,45]})
    fig.subplots_adjust(left=.075, right=.975, top=.91, bottom=.22, wspace=.30)
    for ax, label in zip((a,b), 'ab'):
        decorate(ax); panel(ax, label)
        ax.set_xlabel('Number of events, $T$', labelpad=2)
    readers = ['qwen2.5-7b', 'qwen2.5-14b', 'phi-4']
    for reader, dodge in zip(readers, [-.04, 0., .04]):
        for cond in ['log', 'state']:
            rows = sorted(subset(decay, reader=reader, cond=cond), key=lambda r: int(r['T']))
            x, y = values(rows,'T'), values(rows,'ba')
            a.fill_between(x, values(rows,'lo'), values(rows,'hi'), color=COLORS[reader],
                           alpha=.08, linewidth=0)
            # Only marker glyphs are displaced; trajectories and bands use CSV T.
            a.plot(x, y, color=COLORS[reader],
                   linestyle='-' if cond=='log' else '--', linewidth=1.8)
            a.plot(x * 2**dodge, y, color=COLORS[reader], marker=MARKERS[reader],
                   linestyle='none', markersize=MARKER_SIZES[reader],
                   markeredgewidth=.7, zorder=5,
                   markeredgecolor='white' if cond=='log' else COLORS[reader],
                   markerfacecolor=COLORS[reader] if cond=='log' else 'white')
    ticks = sorted(set(values(decay,'T')))
    log_ticks(a,ticks); a.set_xlim(ticks[0]/1.15, ticks[-1]*1.15)
    a.set_ylim(.4,1.); a.set_ylabel('Balanced accuracy',labelpad=3)
    chance = .5  # Reference specified in the brief, not an observation.
    a.axhline(chance,color='#666666',linestyle=':',linewidth=1.)
    a.text(ticks[0]*1.0375, chance+.015, 'chance',
           ha='left',va='bottom',fontsize=7.5,color='.35')
    reader_legend = a.legend(handles=[reader_handle(r) for r in readers],ncol=3,
                            loc='upper center',bbox_to_anchor=(.49,-.235),columnspacing=.8)
    a.add_artist(reader_legend)
    a.legend(handles=[Line2D([],[],color='#333333',marker='o',markerfacecolor='#333333',label='full log'),
                      Line2D([],[],color='#333333',linestyle='--',marker='o',markerfacecolor='white',label='current state')],
             ncol=2,loc='lower center',bbox_to_anchor=(.48,.035),fontsize=7.5)

    margin = .10  # Registered margin explicitly mandated by the brief.
    b.set_ylim(-.05,.60)
    b.axhline(0,color='.4',linewidth=.65)
    b.axhline(margin,color='#333333',linestyle='--',linewidth=1.2)
    b.text(min(values(horizon,'T'))*1.035,margin+.015,'margin $\\Delta$ = '+f'{margin:.2f}',
           ha='left',va='bottom',fontsize=7.5)
    handles=[]
    for reader in ['qwen2.5-72b','gpt-oss-120b']:
        rows = sorted(subset(horizon,reader=reader,seed=0),key=lambda r:int(r['T']))
        errorbar(b,values(rows,'T'),values(rows,'diff'),values(rows,'lo'),values(rows,'hi'),
                 color=COLORS[reader],marker=MARKERS[reader],markersize=MARKER_SIZES[reader],
                 linewidth=1.9, linestyle='-' if reader=='qwen2.5-72b' else '-.',label=NAMES[reader])
        handles.append(Line2D([],[],color=COLORS[reader],marker=MARKERS[reader],
                              markersize=MARKER_SIZES[reader],linewidth=1.9,
                              linestyle='-' if reader=='qwen2.5-72b' else '-.',label=NAMES[reader]))
    second = subset(horizon,reader='gpt-oss-120b',seed=1)
    errorbar(b,values(second,'T'),values(second,'diff'),values(second,'lo'),values(second,'hi'),
             color=COLORS['gpt-oss-120b'],fmt='none')
    b.plot(values(second,'T')*2**.04,values(second,'diff'),
           color=COLORS['gpt-oss-120b'],marker=MARKERS['gpt-oss-120b'],
           markersize=MARKER_SIZES['gpt-oss-120b'],markerfacecolor='white',linestyle='none',zorder=5)
    handles.append(Line2D([],[],color=COLORS['gpt-oss-120b'],marker=MARKERS['gpt-oss-120b'],
                          markerfacecolor='white',linestyle='none',markersize=MARKER_SIZES['gpt-oss-120b'],
                          label='gpt-oss-120b, second seed'))
    ticks=sorted(set(values(horizon,'T')))
    log_ticks(b,ticks); b.set_xlim(ticks[0]/1.13,ticks[-1]*1.18)
    b.set_ylim(-.05,.60)
    b.set_ylabel('State − log\n(balanced accuracy)',labelpad=4,linespacing=1.1)
    b.legend(handles=handles,loc='upper left',bbox_to_anchor=(.03,.98),labelspacing=.23)
    export(fig,'fig2_horizon',{'decay_csv_rows':len(decay),'decay_series':6,
           'decay_points':sum(len(subset(decay,reader=r)) for r in readers),
           'horizon_csv_rows':len(horizon),'horizon_series':3,
           'horizon_points':len(horizon),'horizon_connected_points':len(horizon)-len(second)})

if __name__ == '__main__': main()
