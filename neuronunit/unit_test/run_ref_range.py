import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# verify that backend is appropriate before compute job:
plt.clf()
import pdb
import copy
import os

import pickle
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from neuronunit.optimization.exhaustive_search import run_grid, reduce_params, create_grid
from neuronunit.optimization import model_parameters as modelp
from neuronunit.optimization import exhaustive_search as es
from neuronunit.models.NeuroML2 import model_parameters as modelp
from neuronunit.models.NeuroML2 .model_parameters import path_params

from neuronunit.optimization.optimization_management import run_ga
from neuronunit.tests import np, pq, cap, VmTest, scores, AMPL, DELAY, DURATION
import matplotlib.pyplot as plt
from neuronunit.models.reduced import ReducedModel
from itertools import product

import quantities as pq
from numba import jit
from neuronunit import plottools
ax = None
import sys


from neuronunit.optimization.optimization_management import build_grid_wrapper, add_constant
from neuronunit.optimization.exhaustive_search import run_rick_grid

from neuronunit.optimization import get_neab
from neuronunit.plottools import plot_surface as ps
from collections import OrderedDict
import pandas as pd


def calc_us(hof,td,history):
    attr_keys = list(hof[0].dtc.attrs.keys())
    hofs_keyed = {}
    us = {} # GA utilized_space
    stds = {}
    means = {}
    for key in attr_keys:
        temp = [ v.dtc.attrs[key] for k,v in history.genealogy_history.items() ]
        us[key] = ( np.min(temp), np.max(temp))
        stds[key] = np.std(temp)
        means[key] = np.mean(temp)
        hofs_keyed[key] = hof[0].dtc.attrs[key]
    return us,stds,means,hofs_keyed

def get_justas_plot(history):
    
    # try:
    import plotly.plotly as py
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot#, iplot
    import plotly.graph_objs as go
    import cufflinks as cf
    cf.go_offline()
    gr = [ v for v in history.genealogy_history.values() ]
    gr = [ g for g in gr if type(g.dtc) is not type(None) ]
    gr = [ g for g in gr if type(g.dtc.scores) is not type(None) ]
    keys = list(gr[0].dtc.attrs.keys())
    xx = np.array([ p.dtc.attrs[str(keys[0])] for p in gr ])
    yy = np.array([ p.dtc.attrs[str(keys[1])] for p in gr ])
    zz = np.array([ p.dtc.attrs[str(keys[2])] for p in gr ])
    ee = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    #pdb.set_trace()
    # z_data = np.array((xx,yy,zz,ee))
    list_of_dicts = []
    for x,y,z,e in zip(list(xx),list(yy),list(zz),list(ee)):
        list_of_dicts.append({ keys[0]:x,keys[1]:y,keys[2]:z,str('error'):e})

    z_data = pd.DataFrame(list_of_dicts)
    data = [
            go.Surface(
                        z=z_data.as_matrix()
                    )
        ]


 
    layout = go.Layout(
            width=1000,
            height=1000,
            autosize=False,
            title='Sciunit Errors',
            scene=dict(
                xaxis=dict(
                    title=str(keys[0]),
        
                    #gridcolor='rgb(255, 255, 255)',
                    #zerolinecolor='rgb(255, 255, 255)',
                    #showbackground=True,
                    #backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    title=str(keys[1]),
        
                    #gridcolor='rgb(255, 255, 255)',
                    #zerolinecolor='rgb(255, 255, 255)',
                    #showbackground=True,
                    #backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    title=str(keys[2]),
        
                    #gridcolor='rgb(255, 255, 255)',
                    #zerolinecolor='rgb(255, 255, 255)',
                    #showbackground=True,
                    #backgroundcolor='rgb(230, 230,230)'
                ),
                aspectratio = dict( x=1, y=1, z=0.7 ),
                aspectmode = 'manual'
            ),margin=dict(
                l=65,
                r=50,
                b=65,
                t=90
            )
        )
    
    fig = go.Figure(data=data, layout=layout)#,xTitle=str(keys[0]),yTitle=str(keys[1]),title='SciUnitOptimization')
    plot(fig, filename='sciunit-score-3d-surface.html')


def get_tests():
    # get neuronunit tests
    # and select out the tests that are more about waveform shape
    # and less about electrophysiology of the membrane.
    # We are more interested in phenomonogical properties.
    electro_path = str(os.getcwd())+'/pipe_tests.p'
    assert os.path.isfile(electro_path) == True
    with open(electro_path,'rb') as f:
        electro_tests = pickle.load(f)

    electro_tests = get_neab.replace_zero_std(electro_tests)
    electro_tests = get_neab.substitute_parallel_for_serial(electro_tests)
    test, observation = electro_tests[0]
    tests = copy.copy(electro_tests[0][0])
    tests_ = tests[0:5]
    tests_.append(tests[-1])
    return tests_, test, observation

tests_,test, observation = get_tests()

grid_results = {}

def plot_scatter(history,ax,keys):
    pop = [ v for v in history.genealogy_history.values() ]
    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in pop ])
    x = np.array([ p.dtc.attrs[str(keys[0])] for p in pop ])
    if len(keys) != 1:
        y = np.array([ p.dtc.attrs[str(keys[1])] for p in pop ])
        ax.cla()
        ax.set_title(' {0} vs {1} '.format(keys[0],keys[1]))
        ax.scatter(x, y, c=y, s=125)#, cmap='gray')
    return ax

def plot_surface(gr,ax,keys,imshow=False):
    # from https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_4thJuly.ipynb
    # Not rendered https://github.com/russelljjarvis/neuronunit/blob/dev/neuronunit/unit_test/progress_report_.ipynb
    gr = [ g for g in gr if type(g.dtc) is not type(None) ]
    gr = [ g for g in gr if type(g.dtc.scores) is not type(None) ]
    ax.cla()
    gr_ = []
    index = 0
    xx = np.array([ p.dtc.attrs[str(keys[0])] for p in gr ])
    yy = np.array([ p.dtc.attrs[str(keys[1])] for p in gr ])
    zz = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    dim = len(xx)
    if imshow==False:
        #ax.pcolormesh(X, Y, Z, edgecolors='black')
        ax.tripcolor(xx,yy,zz)
        # trip_axis = ax_trip.tripcolor(xs,ys,sums,20,norm=matplotlib.colors.LogNorm())
    else:
        import seaborn as sns; sns.set()
        ax = sns.heatmap(Z)

    ax.set_title(' {0} vs {1} '.format(keys[0],keys[1]))
    return ax

def plot_line_ss(gr,ax,key,hof):
    ax.cla()
    ax.set_title(' {0} vs  score'.format(key[0]))
    
    z = np.array([ np.sum(list(p.dtc.scores.values())) for p in gr ])
    x = np.array([ p.dtc.attrs[key[0]] for p in gr ])
    y = hof[0].dtc.attrs[key[0]]
    i = hof[0].dtc.get_ss()
    ax.scatter(x,z)
    ax.scatter(y,i)
    ax.plot(x,z)
    ax.set_xlim(np.min(x),np.max(x))
    ax.set_ylim(np.min(z),np.max(z))
    return ax

def plot_agreement(ax,gr,hof):
    
    dtcpop = [ g.dtc for g in gr ]
    for dtc in dtcpop:
        if hasattr(score,'prediction'):
            if type(score.prediction) is not type(None):
                dtc.score[str(t)][str('prediction')] = score.prediction
                dtc.score[str(t)][str('observation')] = score.observation
                boolean_means = bool('mean' in score.observation.keys() and 'mean' in score.prediction.keys())
                boolean_value = bool('value' in score.observation.keys() and 'value' in score.prediction.keys())

            if boolean_means:
                dtc.score[str(t)][str('agreement')] = np.abs(score.observation['mean'] - score.prediction['mean'])

            if boolean_value:
                dtc.score[str(t)][str('agreement')] = np.abs(score.observation['value'] - score.prediction['value'])

    ss = hof[0].dtc.score
    #for v in ss:
    if str('agreement') in ss.keys():
        ax.plot( [ v['agreement'] for v in list(ss.values()) ], [ i for i in range(0,len(ss.values())) ] )
        ax.plot( [ v['prediction'] for v in list(ss.values()) ], [ i for i in range(0,len(ss.values())) ] )
        ax.plot( [ v['observation'] for v in list(ss.values()) ], [ i for i in range(0,len(ss.values())) ] )
    return ax

def transdict(dictionaries):
    #from collections import OrderedDict
    mps = OrderedDict()
    sk = sorted(list(dictionaries.keys()))
    for k in sk:
        mps[k] = dictionaries[k]
    tl = [ k for k in mps.keys() ]
    return mps, tl

def grids(hof,tests,params,us,history):
    '''
    Obtain using the best candidate Gene (HOF, NU-tests, and expanded parameter ranges found via
    exploring extreme edge cases of parameters

    plot a error surfaces, and cross sections, about the optima in a 3by3 subplot matrix.

    where, i and j are indexs to the 3 by 3 (9 element) subplot matrix,
    and `k`-dim-0 is the parameter(s) that were free to vary (this can be two free in the case for i<j,
    or one free to vary for i==j).
    `k`-dim-1, is the parameter(s) that were held constant.
    `k`-dim-2 `cpparams` is a per parameter dictionary, whose values are tuples that mark the edges of (free)
    parameter ranges. `k`-dim-3 is the the grid that results from varying those parameters
    (the grid results can either be square (if len(free_param)==2), or a line (if len(free_param==1)).
    '''
    temp = OrderedDict(hof[0].dtc.attrs).keys()
    td = list(temp)

    dim = len(hof[0].dtc.attrs.keys())
    flat_iter = iter([(i,freei,j,freej) for i,freei in enumerate(hof[0].dtc.attrs.keys()) for j,freej in enumerate(hof[0].dtc.attrs.keys())])
    #matrix = [[[0 for z in range(dim)] for x in range(dim)] for y in range(dim)]
    plt.clf()
    fig0,ax0 = plt.subplots(dim,dim,figsize=(10,10))
    fig1,ax1 = plt.subplots(dim,dim,figsize=(10,10))

    cnt = 0
    temp = []
    loc_key = {}

    #free_param =
    for k,v in hof[0].dtc.attrs.items():
        loc_key[k] = hof[0].dtc.attrs[k]
        if float(loc_key[k]) != 0.0:
            params[k] = ( loc_key[k]- 2*np.abs(loc_key[k]), loc_key[k]+2*np.abs(loc_key[k]) )
        else:
            params[k] = (-1.0 , 1.0)

    for i,freei,j,freej in flat_iter:
        free_param = [freei,freej]
        free_param_set = set(free_param) # construct a small-set out of the indexed keys 2. If both keys are
        # are the same, this set will only contain one index
        bs = set(hof[0].dtc.attrs.keys()) # construct a full set out of all of the keys available, including ones not indexed here.
        diff = bs.difference(free_param_set) # diff is simply the key that is not indexed.
        # hc is the dictionary of parameters to be held constant
        # if the plot is 1D then two parameters should be held constant.
        hc =  {}
        for d in diff:
            hc[d] = hof[0].dtc.attrs[d]

        cpparams = {}
        if i == j:

            assert len(free_param_set) == len(hc) - 1
            assert len(hc) == len(free_param_set) + 1
            # zoom in on optima


            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))


            #means = { str(freei):hofs_keyed[freei] }
            #std = { str(freei):stds[freei] }

            us,stds,means,hofs_keyed = calc_us(hof,td,history)
            
            rg = build_grid_wrapper(means,stds=stds,k=10)
            _, td = transdict(means)
            td,rg = add_constant(hc,rg,td)

            gr = run_rick_grid(rg, tests,td)
            #us,stds,means,hofs_keyed = calc_us(hof,td,history)

            #gr = run_grid(10,tests,provided_keys = freei, hold_constant = hc,mp_in = params)
            # make a psuedo test, that still depends on input Parametersself.
            # each test evaluates a normal PDP.
            fp = list(copy.copy(free_param))

            #ax1[i,j] = plot_agreement(ax1[i,j],gr,hof)    
            ax0[i,j] = plot_line_ss(gr,ax0[i,j],fp,hof)
            #plot_line_ss(gr,ax1[i,j],fp,hof)
        if i >j:
            assert len(free_param) == len(hc) + 1
            assert len(hc) == len(free_param) - 1

            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))
            cpparams['freej'] = (np.min(params[freej]), np.max(params[freej]))

            us,stds,means,hofs_keyed = calc_us(hof,td,history)
            
            #means = { str(freei):hofs_keyed[freei], str(freej):hofs_keyed[freej] }
            #std = {str(freei):stds[freei], str(freej):stds[freej] }


            rg = build_grid_wrapper(means,stds=stds,k=3)
            _, td = transdict(means)
            td,rg = add_constant(hc,rg,td)


            #td,rg = add_constant(hc,rg,td)

            gr = run_rick_grid(rg, tests,td)

            #gr = run_grid(10,tests,provided_keys = list((freei,freej)), hold_constant = hc, mp_in = params)
            fp = list(copy.copy(free_param))
            ax0[i,j] = plot_surface(gr,ax0[i,j],fp,imshow=False)
            #ax1[i,j] = plot_surface(gr,ax1[i,j],fp,imshow=False)
            #ax1[i,j] = plot_agreement(ax1[i,j],gr,hof):
    
        if i < j:
            free_param = list(copy.copy(list(free_param)))
            if len(free_param) == 2:
                ax0[i,j] = plot_scatter(history,ax0[i,j],free_param)
                ax1[i,j] = ps(fig1,ax1[i,j],freei,freej,history)

            cpparams['freei'] = (np.min(params[freei]), np.max(params[freei]))
            cpparams['freej'] = (np.min(params[freej]), np.max(params[freej]))
            gr = hof

        limits_used = (us[str(freei)],us[str(freej)])
        scores = [ g.dtc.get_ss() for g in gr ]
        params_ = [ g.dtc.attrs for g in gr ]

        # To Pandas:
        # https://stackoverflow.com/questions/28056171/how-to-build-and-fill-pandas-dataframe-from-for-loop#28058264
        temp.append({'i':i,'j':j,'free_param':free_param,'hold_constant':hc,'param_boundaries':cpparams,'scores':scores,'params':params_,'ga_used':limits_used,'grids':gr})
        print(temp)
        #intermediate = pd.DataFrame(temp)
        with open('intermediate.p','wb') as f:
            pickle.dump(temp,f)

    #df = pd.DataFrame(temp)
    plt.savefig(str('cross_section_and_surfaces.png'))
    return temp


opt_keys = [str('vr'),str('a'),str('b')]
nparams = len(opt_keys)
try:
    with open('ranges.p','rb') as f:
        [fc,boundaries] = pickle.load(f)


except:
    # algorithmically find the the edges of parameter ranges, via a course grained
    # sampling of extreme parameter values
    # to find solvable instances of Izhi-model, (models with a rheobase value).
    import explore_ranges


    fc, mp = explore_ranges.pre_run(tests_,opt_keys)
    with open('ranges.p','wb') as f:
        pickle.dump([fc,mp],f)

# get a genetic algorithm that operates on this new parameter range.
try:
    with open('package.p','rb') as f:
        package = pickle.load(f)
        if type(package) is type(list):
            results = {'pop':package[0],'hof':package[1],'pf':package[2],'log':package[3],'history':package[4],'td':package[5],'gen_vs_pop':package[6]}
        else:
            results = package


except:
    exponent = len(opt_keys)
    MU = 2**exponent
    results, DO = run_ga(boundaries,MU,tests_,provided_keys = opt_keys)

    hof = results['hof']
    history = results['history']
    td = results['td']

    us,stds,means,hofs_keyed = calc_us(hof,td,history)
    
    
    boundaries = {k:[hofs_keyed[k]-2*hofs_keyed[k],hofs_keyed[k]+2*hofs_keyed[k]] for k,v in means}
    results, DO = run_ga(boundaries,MU,tests_,provided_keys = opt_keys)

    with open('package.p','wb') as f:
        pickle.dump(results,f)

hof = results['hof']
history = results['history']
td = results['td']
try:
    get_justas_plot(history)
except:
    print('plotly and cufflinks not installed')
us,stds,means,hofs_keyed = calc_us(hof,td,history)
hof = results['hof']
history = results['history']
td = results['td']
#us,stds,means,hofs_keyed = calc_us(hof,td,history)
    


print(means,'means')
print(hofs_keyed,'hofs_keyed')



try:
    assert 1==2
    with open('surfaces.p','rb') as f:
        temp = pickle.load(f)

except:
    #grids(hof,tests,params,us,history):
   
    temp = grids(hof,tests_,boundaries,us,history)
    with open('surfaces.p','wb') as f:
        pickle.dump(temp,f)

sys.exit()
