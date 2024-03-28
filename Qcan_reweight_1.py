import ownmodules as own 
import HS_modules as  HS
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import interp1d
import scipy.optimize 
from scipy.signal import savgol_filter
import random

import math

plt.style.use('../Plot_style/style.txt')
##!!!!!!!!!!!!!! WARNING : this code is an absolute mess !!!!!!!!!!!!!!! 
# Getting better 
R=1.987*10**(-3)
plt.rcParams["font.family"]="Times New Roman"
def plot_legend(Q,N):
    plt.close() 
    colors=[cm.jet_r(float(i)/len(Q)) for i in range(len(Q))] 
    for i in range(len(Q)): 
        plt.scatter([0],[i],color=[colors[i]])
        plt.text(0.01,i,r'$\mathbf{\mathrm{'+str(Q[i])+'_{'+str(N[i])+'}}}$',va='center')
       
    plt.axis('off')
    plt.xlim(-0.01,0.1)
    plt.savefig('legend.pdf')    

def plot_CDF(Proba,Proba_err,Mapping,pH,title=''):
    Proba=np.array(Proba)
    plt.clf()
    fig=plt.figure()
    fig.set_size_inches(8,4)
    ax1=fig.add_axes([0.08,0.11,0.8,0.80])
    colors=[cm.jet_r(float(i)/len(Proba)) for i in range(len(Proba))]
    for i in range(len(Mapping)):
        CDF=np.zeros((len(pH)))
        for p in range(len(pH)):
            CDF[p]=sum(Proba[Mapping[i][0]][:p])
            Proba[Mapping[i][0]][p]
        CDF[:]=CDF[:]/CDF[-1] 
        plt.plot(pH,CDF,color=colors[i])

    plt.xlim(0,14) 
    plt.savefig('CDF_'+title+'.pdf') 

    plt.close()
    plt.xlim(0,14)

    cum_val=np.zeros((len(pH)))
    for i in range(len(Mapping)):
        plt.fill_between(pH,cum_val,cum_val+Proba[i],color=colors[i])    
        cum_val+=Proba[i] 
        plt.plot(pH,cum_val,color='k',linestyle='--',linewidth=0.2)
    plt.xlabel('pH')
    plt.ylabel('Fraction of the ensemble')
    plt.ylim(0,1)
    plt.savefig('Meso_cumul_'+title+'.pdf')
    plt.close()

    include=[]
    legend_save=[]      
    colors=['r','g']
    fractions=[0.95,0.99]
    plt.close()
    fig=plt.figure()
    fig.set_size_inches(8,6)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    for j in range(len(fractions)):
        include+=[[]]  
        fraction_current=np.zeros((len(pH)))
        out_N=np.zeros((len(pH)))
        for p in range(len(pH)):
            temp=np.flip(np.sort(Proba[:,p]))
            for i in range(len(Proba)):         
                if fraction_current[p]<fractions[j]: 
                    fraction_current[p]+=temp[i]
                    out_N[p]+=1
                    if pH[p]<7.8 and pH[p+1]>=7.8 :
                        include[j]+=[i]

        txt='Threshold='+str(fractions[j])
        ax.plot(pH,out_N,color=colors[j],label=txt)
    ax.legend(loc=1,prop={'size': 10})
    plt.xlim(0,14) 
    plt.ylabel('Number of mesostates')
    plt.xlabel('pH')
    plt.savefig('Meso_N_'+title+'.pdf')

def wrap_plot(F_fit_man,title,pKas_plot=[]):
    F_new=np.array([[F_fit_man[i]] for i in range(len(F_fit_man))])
    Fs_err=F_new*0.
    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(F_new,Fs_err,pH,T,ign_norm_err=True,unsafe=0)
    Mapping=[[i] for i in range(len(F_new))]
    plot_probas_spec(Proba,Proba_err,Mapping,pH,pKas=pKas_plot,title=title)
    pH_extended=np.linspace(-3,17,200)
    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(F_new,Fs_err,pH_extended,T,ign_norm_err=True,unsafe=0)
    plot_CDF(Proba,Proba_err,Mapping,pH_extended,title=title)        

def plot_probas_spec(Proba,Proba_err,Mapping,pH,title='Proba',subrep='./',pKas=[],separate=0,print_micro=1,do_layer=1,Q_min_max=[]):
    import matplotlib
    from matplotlib import cm
    import matplotlib.pyplot as plt
#    plt.switch_backend('TkAgg')   
    plt.rcParams["font.family"]="Times New Roman"
    #matplotlib.use("pgf")
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
    matplotlib.rcParams['text.latex.unicode']=False
    
    #matplotlib.rcParams["text.latex.preamble"].append(r'\usepackage[dvips]{graphicx}\usepackage{xfrac}')
    pgf_with_latex = {
        "text.usetex": True,
        "pgf.preamble": [
            r'\usepackage{color}' ],
        "font.family": "Times New Roman"
    }

    import matplotlib
    matplotlib.rcParams.update(pgf_with_latex)
    import matplotlib.pyplot as plt
    from matplotlib import cm

    colors=[cm.jet_r(float(i)/len(Proba)) for i in range(len(Proba))]

    if separate==0 and print_micro==1:
        fig=plt.figure()
        fig.set_size_inches(8,4)
        ax1=fig.add_axes([0.08,0.125,0.8,0.80])
        ticks_2=[]
        
        for i in range(len(pKas)):    
            ticks_2+=[str(round(pKas[i],2))]
            #plt.text(pKas[i],0.8,'pH='+str(pKas[i]),rotation=45) 
            plt.vlines(round(pKas[i],2),0,1,linestyle='--',color='grey')

        ticks_2+=[str(7)] 
        pKas+=[7.]
        axT=ax1.twiny()
        axT.set_xticks(pKas) 
        axT.set_xticklabels(ticks_2)
        plt.xlim(0,14)
        plt.sca(ax1) 
        plt.xlim(0,14) 
 
        plt.vlines(7,0,1,color='m',alpha=0.5)
        
        plt.ylim(0,1)
        
        plt.xlabel("pH")
        plt.ylabel("Mesostate probability")
        for i in range(len(Proba)):
            low_err=[]                 
            high_err=[]
            if Proba_err[i]!=[]: 
                for p in range(len(Proba[i])):
                    if math.isinf(Proba_err[i][p]):
                        low_err+=[Proba[i][p]-Proba_err[i][p]]
                        high_err+=[Proba[i][p]+Proba_err[i][p]]
                    else :
                        low_err+=[Proba[i][p]]
                        high_err+=[Proba[i][p]]
            else :
                low_err+=Proba[i]
                high_err+=Proba[i]
            if len(Proba[i])==0 :
                continue
            plt.plot(pH,Proba[i],color=colors[i],linestyle='-')
#            plt.fill_between(pH, low_err, high_err, color=colors[i],alpha=0.5)
            #ax1.set_rasterized(True)  
        if do_layer==1 :
            # Plotting the mesostates
            Proba_cum=[]
            for i in range(len(Mapping)):
                temp=[0 for p in range(len(pH))]
                for j in range(len(Mapping[i])):
                    for p in range(len(pH)):
                        temp[p]+=Proba[int(Mapping[i][j])][p]

                Proba_cum+=[temp]
            for i in range(len(Proba_cum)):
                ax1.plot(np.array(pH),np.array(Proba_cum[i]),color='k',linestyle='--',linewidth=0.3)
                #ax1.set_rasterized(True)

#        ax2=fig.add_axes([0.88,0.05,0.2,0.90])
#        for i in range(len(Proba)): 
#            ax2.scatter([0],[i],color=[colors[i]])
#            #temp_txt='' 
#        plt.axis('off')
#        plt.xlim(-0.01,0.1)
        
        FU.check_and_create_rep('Results/Plots/Detailed')
        FU.check_and_create_rep('Results/Plots/Detailed/'+subrep)
        plt.savefig('Results/Plots/Detailed/'+subrep+'/States_'+title+'.pdf')
        plt.clf()


if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps","-ns",help='Number of steps in the fitting')
    parser.add_argument("--num_restarts","-nr",help='Number of restarts in the fitting')
    parser.add_argument("--verbose","-v",help='Frequency of output during fit.')
    parser.add_argument('--gradual',"-g", default=False, action='store_true')
    parser.add_argument('--plot_only',"-po", default=False, action='store_true')
    parser.add_argument('--use_man',"-um", default=False, action='store_true')
    parser.add_argument('--raw_only',"-ro", default=False, action='store_true')
    parser.add_argument('--pre_fit',"-pf", default=False, action='store_true')
    args = parser.parse_args() 
    # The strategy here is to fit the data with the original pKas
    
    if args.num_steps :
        try : 
            num_steps=int(args.num_steps)
        except :
            num_steps=100000
            print("Invalid input for num_steps, defaulting to ",num_steps)
    else : 
        num_steps=100000
        print("No input for num_steps, defaulting to ",num_steps)

    if args.num_restarts :
        try : 
            N_restarts=int(args.num_restarts)
        except :
            N_restarts=10
            print("Invalid input for num_restarts, defaulting to ",N_restarts)
    else : 
        N_restarts=10
        print("No input for num_restarts, defaulting to ",N_restarts)

    if args.verbose :
        try : 
            v_freq=int(args.verbose)
        except :             
            v_freq=10000
            print("Invalid input for verbose, defaulting to ",v_freq)
    else : 
        v_freq=10000
        print("No input for verbose, defaulting to ",v_freq)

    use_man_fit=args.use_man # True #False
    plot_only=args.plot_only #False#True
    raw_only=args.raw_only #True
    no_lin=not args.pre_fit #False
    pH_res=100
    window_size=101
    poly_order=2

    T=QC.read_param()[0]
    power_diff=1
    restart_prec_frac=0.25# This is the tolerance compared to the  best fit so far
    T_acc=0.003
    rand_mag=0.01 #0.01#0.001
    target_prec=0.0

    max_unchanged=20000
    temp_check_freq=500 #How often to check the acceptance ratio to adapt temp
    temp_factor=0.01 #How much to change temperature
    target_acc=0.5#Original target acceptance ratio
    gradual=args.gradual#True
    #Number of restarts    
    rand_only=True 

    fit_params=[T,power_diff,restart_prec_frac,T_acc,rand_mag,target_prec,v_freq,num_steps,max_unchanged,temp_check_freq,temp_factor,target_acc,gradual,N_restarts,rand_only]

    data=FU.read_file('folders.txt')
    loc_org=data[0][1]+'/'
    loc_hel=data[1][1]+'/'
    loc_dis=data[2][1]+'/'
    seq_3=QC.convert_residues_to_titrable(loc_org+'seq.in')
    seq_1=QC.convert_AA_3_to_1_letter(seq_3)
    list_res=[i-1 for i in range(len(seq_3))]
    pos_res,neg_res,base_charge,arg_res,raw_seq,seq_data_q,seq_data_id,seq_id_reduced,map,new_W=QC.HSQ_internal_sequence_create(seq_3,list_res)
   
    target_Q=np.array(FU.read_file('q_vs_pH.txt'),dtype=np.float)

#    CD_data=np.array(np.array(FU.read_file('CD_vs_pH.txt'))[1:],dtype=float)
#    plt.scatter(CD_data[:,0],CD_data[:,1]/CD_data[:,2])
#    plt.savefig('test_SSP.png')
#    plt.clf()
    
    #Not sure wht those are for 
#    try :
#        Fs_org=QC.read_HSQ_format(loc_org+'Results/Fs/Normal/F_per_state.txt',False)
#    except : 
#        print "0"
#    Fs_hel=QC.read_HSQ_format(loc_hel+'Results/Fs/Normal/F_per_state.txt',False)
#    Fs_dis=QC.read_HSQ_format(loc_dis+'Results/Fs/Normal/F_per_state.txt',False)
#
    F_per_lvl_pred=FU.read_file(loc_org+'Results/Fs/Global_F_prediction.txt')
    Fs_pred=np.array(F_per_lvl_pred,dtype=float)[:,1]

    #layers_q=FU.read_file(loc_org+'/Charge_layers.txt')
    
#    q=np.array(layers_q,dtype=int)[:,0]

    q=[i for i in range(neg_res,pos_res+arg_res)]

    Ni=len(seq_1)-5
    pH=np.linspace(0,14,14*pH_res)
    
    N_1=pos_res-arg_res
    N_2=0
    N_3=neg_res#Ni/2
    
    #pKas=np.array([10.5 for k in range(N_1)]+[4.2 for k in range(N_3)])

    pKas=np.array([10.5 for k in range(N_1)]+[4.2 for k in range(N_3)])

    #pKas=np.array([7. for k in range(N_1)]+[7. for k in range(N_3)])

    if len(q)==9:
        q_offset=4#4.34
        pH_min=2.7
        pH_max=12.35#11.3 #11.78#3
        mean_1=15
        mean_2=7
        mean_3=4.85
        std_1=4.
        std_2=1
        std_3=5
        rescale_factor=0.92##0.88#2
    elif len(q)==17:
        q_offset=8#10.55#5#8
        pH_min=2.7
        pH_max=11.1
        mean_1=12.9
        mean_2=7
        mean_3=4.5
        std_1=4.15
        std_2=1
        std_3=2.8
        rescale_factor=0.76#0.75#8
    elif len(q)==25:
        q_offset=12#15.2 
        pH_min=2.5
        pH_max=11.3
        mean_1=13.#14.05
        mean_2=7
        mean_3=5.5#4.75
        std_1=0.1
        std_2=1
        std_3=3.1
        rescale_factor=0.79

    Fake_F=np.zeros((len(Fs_pred)))

    W=''
    for l in range(1,len(Fs_pred)):
        Fake_F[l]=(Fake_F[l-1]-all_DFs[l-1])
    
    Fake_F=Fake_F[:]+abs(min(Fake_F))
    for l in range(len(Fake_F)):
        W+=str(Fake_F[l])+'\n'

    FU.write_file('./Results/Fs/Global_F_per_level.txt',W)    
    Fs=all_DFs

    ind_min=0
    ind_max=-1        
    plt.clf()

    target_Q[:,1]=rescale_factor*target_Q[:,1]+q_offset
    plt.scatter(target_Q[:,0],target_Q[:,1])
    plt.ylabel('Protein net charge')
    plt.xlabel('pH')
    plt.vlines([pH_min,pH_max],np.amin(target_Q[:,1]),np.amax(target_Q[:,1]),linestyle='--',color='grey')
    plt.xlim(0,14)
    plt.hlines([q[0],q[-1]],0,14,linestyle='--',color='grey')
    plt.savefig('raw_data.pdf')
    for k in range(len(target_Q[:,0])):
        if target_Q[k,0]>pH_min and target_Q[k-1,0]<=pH_min :
            ind_min=k
        elif target_Q[k,0]>pH_max and target_Q[k-1,0]<=pH_max :
            ind_max=k

    q_out=QC.get_q_profile_from_F(pH,Fake_F,q,T)

    ind_min_pH=0
    ind_max_pH=-1

    for k in range(len(pH)):
        if pH[k]>pH_min and pH[k-1]<=pH_min :
            ind_min_pH=k                
        elif pH[k]>pH_max and pH[k-1]<=pH_max :
            ind_max_pH=k
 
    if raw_only==True : 
        raw_data=target_Q[ind_min:ind_max,1]
        raw_pH=target_Q[ind_min:ind_max,0]
    else :  
        raw_data=np.append(q_out[:ind_min_pH],target_Q[ind_min:ind_max,1])
        raw_data=np.append(raw_data,q_out[ind_max_pH:])
        raw_pH=np.append(pH[:ind_min_pH],target_Q[ind_min:ind_max,0])
        raw_pH=np.append(raw_pH,pH[ind_max_pH:])
    
    def linearize_data(data,scale,new_scale):
        new_data=np.zeros((len(new_scale)))
        for j in range(len(scale)-1):
            for i in range(len(new_scale)):
        	    if new_scale[i]>=scale[j] and new_scale[i]<=scale[j+1] : 
        	        slope=(data[j+1]-data[j])/(scale[j+1]-scale[j])
        	        diff_pH=new_scale[i]-scale[j]
        	        new_data[i]=data[j]+diff_pH*slope		
        		
        return new_data
    
    plt.ylabel('Protein net charge') 
    plt.xlabel('pH')
    plt.hlines(0,0,14)
    plt.plot(pH,q_out)
    plt.scatter(raw_pH,raw_data)
    plt.savefig('test_prelin.pdf')	
    plt.clf()

    d1_prefit=QC.get_first_derivative(pH,q_out)       

    new_pH=np.linspace(raw_pH[0],raw_pH[-1],(raw_pH[-1]-raw_pH[0])*pH_res)
    ind_min_new_pH=0
    ind_max_new_pH=-1

    for k in range(len(new_pH)):
        if new_pH[k]>pH_min and new_pH[k-1]<=pH_min :
            ind_min_new_pH=k
        elif new_pH[k]>pH_max and new_pH[k-1]<=pH_max :
            ind_max_new_pH=k

    itp = interp1d(raw_pH,raw_data, kind='linear')
    yy_sg = savgol_filter(itp(new_pH), window_size, poly_order)
    
    N=0
    val=0.
    for p in range(len(raw_pH)):
        ind=QC.find_index(new_pH,raw_pH[p])
        N+=1
        val+=(raw_data[p]-yy_sg[ind])**2
    
    new_data=yy_sg
    new_DF=np.zeros((len(q)))

    fig=plt.figure()
    fig.set_size_inches(7.5,5)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
    plt.xlim(0,14)
    plt.ylabel('Protein net charge')
    plt.xlabel('pH')

    plt.scatter(target_Q[:,0],target_Q[:,1],color='grey')
    plt.scatter(raw_pH,raw_data,color='orange')
    plt.plot(new_pH,new_data,color='r')
    plt.savefig('./smoothed_curve.pdf')

    plt.close()
    fig=plt.figure()
    fig.set_size_inches(7.5,5)
    ax=fig.add_axes([0.1,0.1,0.8,0.8])
        
    d1_data=QC.get_first_derivative(raw_pH,raw_data)
 
    plt.plot(raw_pH,d1_data,color='orange')

    # Now the pre derivative
    d1_new_data=QC.get_first_derivative(new_pH,new_data)
    plt.plot(new_pH,d1_new_data,color='r')
    
    plt.vlines([pH_min,pH_max],q[0],0)
    plt.savefig('prelin_deriv.pdf')                                    
    plt.clf()
    
    for p in range(1,len(new_pH)):
        for q1 in range(len(q)-1):
            a=(q[q1] +q[q1+1])/2.
            if new_data[p-1]>a and new_data[p]<=a:
                new_DF[q1]=R*T*np.log(10.)*new_pH[p]
        
    new_Fs=np.array([sum(new_DF[k:]) for k in range(len(new_DF))]+[0.])

    def find_index(arr,val):
        for i in range(1,len(arr)):
            if (arr[i-1]<val and arr[i]>=val) or ((arr[i-1]>val and arr[i]<=val)) :
                return i
    
    q_out=QC.get_q_profile_from_F(new_pH,Fake_F,q,T)

    q_out_org=q_out.copy()
    sum_temp=0
    Fake_F_org=Fake_F.copy()
    
    F_pre_fit=np.zeros((len(pKas)+1))
    DF=np.zeros((len(pKas)))
    for u in range(len(pKas)):
        DF[u]=float(pKas[u])*R*T*np.log(10.)                                            
    for u in range(len(DF)):
        F_pre_fit[u]=np.sum(DF[u:])
     
#    F_pre_fit=Fs_pred.copy()
#    F_pre_fit=Fake_F.copy()
    if plot_only==False:    
        if no_lin==True :
            Fake_F_all=QC.fit_potentiometric_data_V1(F_pre_fit,raw_data,raw_pH,q,fit_params) 
        else : 
            Fake_F_all=QC.fit_potentiometric_data_V1(F_pre_fit,new_data,new_pH,q,fit_params)
        diff_fit_temp=float('+inf')                                                   
        save_ind=0
        for k in range(len(Fake_F_all)):
            q_out=QC.get_q_profile_from_F(new_pH,Fake_F_all[k],q,T)
            diff_fit=sum((abs(q_out-new_data)**power_diff)/len(pH))
            if diff_fit<diff_fit_temp:
                save_ind=k
                diff_fit_temp=diff_fit 
        Fake_F=Fake_F_all[save_ind]
    else : 
        data=FU.read_file('pKas_in.txt')
        pKas=[float(data[l][0]) for l in range(len(data))]
        Fake_F=np.zeros((len(pKas)+1))
        DF=np.zeros((len(pKas)))
        for u in range(len(pKas)):
            DF[u]=float(pKas[u])*R*T*np.log(10.)                                        
        for u in range(len(DF)):
            Fake_F[u]=np.sum(DF[u:])        
    
    q_out_pred=QC.get_q_profile_from_F(pH,Fs_pred,q,T)
    
    
    
    if use_man_fit:
        data=FU.read_file('pKas_man.txt')
        F_fit_man=np.zeros((len(DF)+1))
        for i in range(len(DF)):
            F_fit_man[i]=np.sum(DF[i:])
        q_out_fit=QC.get_q_profile_from_F(pH,F_fit_man,q,T)
    else : 
        q_out_fit=QC.get_q_profile_from_F(pH,Fake_F,q,T)

    Fake_F_simple=[0]
    for k in range(len(Fake_F)/2):
        Fake_F_simple+=[Fake_F_simple[-1]+4.34*np.log(10)*R*T]
    for k in range(len(Fake_F)/2,len(Fake_F)-1):
        Fake_F_simple+=[Fake_F_simple[-1]+10.34*np.log(10)*R*T]
    Fake_F_simple=np.flip(np.array(Fake_F_simple))
 
    plateau_data=[]
    plateau_pH=[]
    for i in range(len(new_pH)):
        if new_pH[i]<pH_min:
            plateau_data+=[new_data[i]]
            plateau_pH+=[new_pH[i]]
        elif new_pH[i]>pH_max:
            plateau_data+=[new_data[i]]
            plateau_pH+=[new_pH[i]]
   
    if use_man_fit==True :
        wrap_plot(F_fit_man,'Fit_man')
    else :
        wrap_plot(Fake_F,'Fit',pKas_plot=[pKas[len(pKas)/2-1],pKas[len(pKas)/2]])
    wrap_plot(Fs_pred,'Unshifted')
    wrap_plot(Fake_F_simple,'Simple')

     
    N=[int(scipy.misc.comb(len(q)/2,r)) for r in range(len(q)/2)]
    N=N+N+[1]
    
    plot_legend(q,N)
    q_out_simple=QC.get_q_profile_from_F(pH,Fake_F_simple,q,T)
    plt.close()     

    q_quen=[q[0],q[len(q)/2],q[-1]] 
    F_Fake_quen=[Fs_pred[0],Fs_pred[len(q)/2],Fs_pred[-1]]
    q_out_quen=QC.get_q_profile_from_F(pH,F_Fake_quen,q_quen,T)

#    plt.vlines(pKas,q[-1],q[0],color='gray',linestyle='--')
    plt.hlines(0,0,14,color='gray',linestyle='--')
    plt.scatter(target_Q[ind_min:ind_max,0],target_Q[ind_min:ind_max,1],color='orange',label='Used experimental data')
    plt.scatter(target_Q[ind_max:,0],target_Q[ind_max:,1],color='grey',label='Unused experimental data')
    plt.scatter(target_Q[:ind_min,0],target_Q[:ind_min,1],color='grey')
    plt.plot(pH,q_out_quen,color='k',label="Fixed charge model")
    plt.plot(pH[ind_min_pH:ind_max_pH],q_out_fit[ind_min_pH:ind_max_pH],color='b',label="Fit to equation (9)")
    plt.plot(pH[:ind_min_pH],q_out_fit[:ind_min_pH],color='b',linestyle='--')
    plt.plot(pH[ind_max_pH:],q_out_fit[ind_max_pH:],color='b',linestyle='--') 

    plt.ylim(q[0]-1,q[-1]+1)        
    plt.ylabel('Protein net charge')
    plt.xlabel('pH')
    plt.legend(loc=1,prop={'size': 10})
    plt.savefig('q_vs_pH_fit_quen.pdf')
    
    plt.close() 
  
    fig=plt.figure()
    fig.set_size_inches(7.5,5)    
    ax1=fig.add_axes([0.09,0.2,0.8,0.70])

    plt.ylim(q[0]-1,q[-1]+1)
    plt.scatter(target_Q[ind_min:ind_max,0],target_Q[ind_min:ind_max,1],color='orange',label='Used experimental data') 
    plt.scatter(target_Q[ind_max:,0],target_Q[ind_max:,1],color='grey',label='Unused experimental data')
    plt.scatter(target_Q[:ind_min,0],target_Q[:ind_min,1],color='grey') 
    #plt.plot(pH,q_out_quen,color='k') 
    plt.plot(pH,q_out_pred,color='r',label='Unshifted pK'+r'$\mathbf{_a}$'+' values') 
    plt.ylabel('Protein net charge')
    plt.xlim(0,14)
    plt.hlines(0,0,14,color='grey',linestyle='--')
    plt.plot(pH[ind_min_pH:ind_max_pH],q_out_fit[ind_min_pH:ind_max_pH],color='b',label='Fit to equation (9)')
    plt.plot(pH[:ind_min_pH],q_out_fit[:ind_min_pH],color='b',linestyle='--')
    plt.plot(pH[ind_max_pH:],q_out_fit[ind_max_pH:],color='b',linestyle='--') 
    ax1.set_xticks([], []) 
    ax1.legend(loc=1,prop={'size': 10})

    ax2=fig.add_axes([0.09,0.11,0.8,0.09])
    plt.hlines(0,0,14,color='grey',linestyle='--') 
    q_out_raw=QC.get_q_profile_from_F(raw_pH,Fake_F,q,T)
    
    plt.xlim(0,14)  
    plt.plot(raw_pH,raw_data-q_out_raw,color='k')
    ax2.yaxis.tick_right()
    plt.ylabel('Residual')
    ax2.yaxis.set_label_position("right")
    plt.xlabel('pH')
    plt.tight_layout()
    plt.savefig('q_vs_pH_fit.pdf')
    plt.close()

    plt.hlines(0,0,14,color='gray',linestyle='--')
    d1_expt=QC.get_first_derivative(new_pH,new_data)
    plt.plot(new_pH,d1_expt,color='orange')
       
    d1_quen=QC.get_first_derivative(pH,q_out_quen)
    plt.plot(pH,d1_quen,color='k')
    plt.savefig('derivative_Q_quenched.pdf')
    
    plt.savefig('derivative_Q_unshifted.pdf') 

    fig=plt.figure()
    fig.set_size_inches(8.5,5)
    ax1=fig.add_axes([0.11,0.2,0.8,0.70])

    plt.ylabel(r'$\frac{dq}{dpH}$')    
    plt.hlines(0,0,14,color='grey',linestyle='--') 
    d1_unsh=QC.get_first_derivative(pH,q_out_pred)         
    plt.plot(pH[ind_min_pH:ind_max_pH],d1_unsh[ind_min_pH:ind_max_pH],color='r',label='Unshifted pK'+r'$\mathbf{_a}$'+' values')
    plt.plot(pH[:ind_min_pH],d1_unsh[:ind_min_pH],color='r',linestyle='--')
    plt.plot(pH[ind_max_pH:],d1_unsh[ind_max_pH:],color='r',linestyle='--')

    d1_expt=QC.get_first_derivative(new_pH,new_data)
    plt.plot(new_pH,d1_expt,color='orange',linewidth=3,label='Smoothed experimental data')

    #plt.plot(new_pH[:ind_min_new_pH],d1_expt[:ind_min_new_pH],color='orange',linestyle='--')
    #plt.plot(new_pH[ind_max_new_pH:],d1_expt[ind_max_new_pH:],color='orange',linestyle='--') 
    #d1_simple=QC.get_first_derivative(pH,q_out_simple)         
    #plt.plot(new_pH,d1_simple,color='g')
    #plt.savefig('derivative_Q_unshifted.pdf')
    plt.xlim(0,14) 
    d1_fit=QC.get_first_derivative(pH,q_out_fit) 
    plt.plot(pH[ind_min_pH:ind_max_pH],d1_fit[ind_min_pH:ind_max_pH],color='b',label="Fit using equation (9)")
    plt.plot(pH[:ind_min_pH],d1_fit[:ind_min_pH],color='b',linestyle='--')
    plt.plot(pH[ind_max_pH:],d1_fit[ind_max_pH:],color='b',linestyle='--') 

    ax1.legend(prop={'size': 8})
    ax1.set_xticks([], [])
    ax2=fig.add_axes([0.11,0.11,0.8,0.09])
    plt.hlines(0,0,14,color='grey',linestyle='--') 

    ax2.yaxis.tick_right()
    plt.ylabel('Residual')
    ax2.yaxis.set_label_position("right")
    plt.xlim(0,14)  
    plt.xlabel('pH') 
    q_out_fit_res=QC.get_q_profile_from_F(new_pH,Fake_F,q,T)
    d1_fit_res=QC.get_first_derivative(new_pH,q_out_fit_res) 
    
    plt.plot(new_pH,d1_fit_res-d1_expt,color='k')
    
    plt.tight_layout()
    plt.savefig('derivative_Q_fit.pdf')

    Fs_err=Fake_F*0.

    Fake_F=np.array([[Fake_F[j]] for j in range(len(Fake_F))])
    pKa=[ (Fake_F[k-1][0]-Fake_F[k][0])/(R*T*np.log(10.)) for k in range(1,len(Fake_F))]
    if plot_only==False : 
        W=''
        for k in range(len(pKa)):
            W+=str(pKa[k])+'\n'
        FU.write_file('pKas_out.txt',W)
       
    #Here is get the hill coefficient fromthe proba
    W,W_err,Proba,Proba_err=QC.get_probas(Fake_F,Fake_F,pH,T,ign_norm_err=True,unsafe=0)
    basic=[i for i in range(len(Fake_F))]
    plt.close()
    N_a=np.zeros((len(pH)))
    N_b=np.zeros((len(pH)))
    for p in range(len(pH)):
        for i in range(len(Proba)):
            
            N_a[p]+=((basic[-1]-basic[i])*Proba[i][p])
            N_b[p]+=basic[i]*Proba[i][p]
    Hp=10**-pH[:]
    Kc=Hp[:]*N_b[:]/N_a[:]
    ln_Kc=np.log(Kc[:])
    ln_Hp=np.log(Hp[:])
    dln_Kc=QC.get_first_derivative(ln_Hp,ln_Kc)
    plt.close()
    h=dln_Kc-1.
    plt.plot(pH,h,color='g')
    plt.xlabel('pH')
    plt.ylabel('Hill coefficient')
    plt.savefig('Hill.pdf')
    plt.close()

    
        
    #plt.vlines(pKas,q[-1],q[0],color='gray',linestyle='--')
    plt.hlines(0,0,14,color='gray',linestyle='--')
    plt.scatter(target_Q[ind_min:ind_max,0],target_Q[ind_min:ind_max,1],color='orange')
    plt.scatter(target_Q[ind_max:,0],target_Q[ind_max:,1],color='grey')
    plt.scatter(target_Q[:ind_min,0],target_Q[:ind_min,1],color='grey')
    plt.plot(pH,q_out_quen,color='k')

    plt.ylim(q[0]-1,q[-1]+1)
    plt.ylabel('Protein net charge')
    plt.xlabel('pH')
    plt.savefig('q_vs_pH_quenched_unshifted.pdf')

    plt.plot(pH,q_out_pred,color='r')
    plt.savefig('q_vs_pH_unshifted.pdf')


    plt.plot(pH,q_out_simple,color='g')
    plt.savefig('q_vs_pH_simple.pdf')   

    for i in range(len(Fake_F_all)):
        q_out_fit=QC.get_q_profile_from_F(pH,Fake_F_all[i],q,T)
        plt.plot(pH[ind_min_pH:ind_max_pH],q_out_fit[ind_min_pH:ind_max_pH],color='b')
        plt.plot(pH[:ind_min_pH],q_out_fit[:ind_min_pH],color='b',linestyle='--')
        plt.plot(pH[ind_max_pH:],q_out_fit[ind_max_pH:],color='b',linestyle='--') 
    plt.savefig('q_vs_pH_fit_all.pdf')
    plt.close()
       

    plt.hlines(0,0,14,color='gray',linestyle='--')
    for i in range(len(Fake_F_all)):
        q_out_fit=QC.get_q_profile_from_F(pH,Fake_F_all[i],q,T)
        d1_fit=QC.get_first_derivative(pH,q_out_fit) 
        plt.plot(pH[ind_min_pH:ind_max_pH],d1_fit[ind_min_pH:ind_max_pH],color='b')
        plt.plot(pH[:ind_min_pH],d1_fit[:ind_min_pH],color='b',linestyle='--')
        plt.plot(pH[ind_max_pH:],d1_fit[ind_max_pH:],color='b',linestyle='--') 
    d1_expt=QC.get_first_derivative(new_pH,new_data)
    plt.plot(new_pH,d1_expt,color='orange')
    plt.savefig('derivative_Q_fit_all.pdf')
