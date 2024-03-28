import Fossat_utils as FU
import qcan_utils as  QC
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.optimize 
from scipy.signal import savgol_filter
import random
from matplotlib import cm
##!!!!!!!!!!!!!! WARNING : this code is an absolute mess !!!!!!!!!!!!!!! 
# Getting better 
R=1.987*10**(-3)

plt.rcParams["font.family"]="Times New Roman"
def get_E_fit(Proba,meso_hel,target):
    E_fit=0
    for p in range(len(target)):
        E_fit+=abs(np.sum(Proba[:,p]*meso_hel[:])-target[p])    
    return E_fit

def print_fig(hel_meso,pH_raw,fraction_hel_used,Fs,title):
    plt.close()
    pH=np.arange(0,14,0.1,dtype=np.float)
    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(Fs,Fs,pH,T,ign_norm_err=True,unsafe=0)
    Proba=np.array(Proba)
    hel_pH=np.zeros((len(pH)))
    for p in range(len(pH)):
        for s in range(len(hel_meso)):
            hel_pH[p]+=Proba[s,p]*hel_meso[s]
    plt.plot(pH,hel_pH)
    plt.scatter(pH_raw,fraction_hel_used)
    plt.ylim(0.,1.)
    plt.savefig(title+'.pdf')

def get_target_per_meso(target,pH,Fs,T): 
    Fs_err=Fs*0.
    T_acc=0.5        
    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(Fs,Fs_err,pH,T,ign_norm_err=True,unsafe=0)
    Proba=np.array(Proba)
    # Okay so now randomly select a state
    #randomly change helicity
    #evaluate cost function
    #accept/reject
    hel_org=np.array([random.random() for i in range(len(Proba))])#np.array([np.mean(target) for i in range(len(Proba))])
    hel_meso=hel_org.copy()
    E_fit_org=get_E_fit(Proba,hel_org,target)
    E_fit=E_fit_org
    E_fit_best=E_fit_org 
    hel_meso_best=hel_meso.copy()
    count=0
    num_acc=0
    full_rand=False
    acc_max=0.95
    acc_min=0.70
    N_restart=num_restart
    max_steps=num_steps
    started=False
    restart=0
    hel_meso_best_save=[]

    while True :  
        hel_meso_new=hel_meso.copy()
        if full_rand: 
            for s in range(len(hel_meso)):
                hel_meso_new[s]=random.random()
        else :
            s=random.randint(0,len(Proba)-1) 
            hel_meso_new[s]=random.random()
        E_fit_new=get_E_fit(Proba,hel_meso_new,target)
        diff=E_fit_new-E_fit
        rand=random.random()        
        exp=np.exp(-(diff)/T_acc)
        acc=exp>rand
        count+=1
        if acc :                
            E_fit=E_fit_new
            hel_meso=hel_meso_new.copy()
            num_acc+=1  
            
        if E_fit<E_fit_best:
            E_fit_best=E_fit
            hel_meso_best=hel_meso.copy()
        
        if started==False and float(count)%500==0:
            grad=0.05
            if num_acc/float(count)>acc_max  :
                print("restarting ",T_acc,num_acc/float(count))
                count=0
                num_acc=0
                T_acc=T_acc*(1-grad)
                continue
            elif num_acc/float(count)<acc_min:
                print("restarting ",T_acc,num_acc/float(count))
                T_acc=T_acc*(1+grad)
                continue
            else :
                started=True

        if float(count)%1000==0:
            if E_fit*0.75>E_fit_best:
                E_fit=E_fit_best
                hel_meso=hel_meso_best
        if float(count)%10000==0:
            print(count)
            print("Acc : ",num_acc/float(count))
            print("Curr fit : ",E_fit)
            print("Best_fit : ",E_fit_best)
            print_fig(hel_meso_best,pH,target,Fs,"Current")
            #Weights_norm,Weights_norm_err,Proba_full,Proba_err=QC.get_probas(Fs,Fs_err,pH_fine,T,ign_norm_err=True,unsafe=0)
            #print_fig(Proba_full,hel_meso,pH)
        if count>max_steps and restart<N_restart: 
            print("Restarting ("+str(restart)+" out of "+str(N_restart)+")") 
            count=0
            loc='Fits_details/Round_'+str(restart)
            FU.check_and_create_rep(loc)
            print_fig(hel_meso_best,pH,target,Fs,loc+"/Best_fit_"+str(restart))

            W='' 
            for s in range(len(hel_meso_best)):
                W+=str(q[s])+'\t'+str(hel_meso_best[s])+'\n'
            FU.write_file(loc+'/Hel_per_meso.txt',W) 
            hel_meso_best_save+=[hel_meso_best]
            hel_meso_best=np.array([random.random() for i in range(len(Proba))]) 
            hel_meso=hel_meso_best.copy()
            E_fit=float('+inf')
            restart+=1
            E_fit_best=E_fit
        elif count>max_steps:
            break

    print(num_acc/float(count)) 
    return hel_meso_best_save

def plot_hist_mix(vals,title):
    plt.close()
    vals=np.array(vals) 
    colors=[cm.jet_r(float(i)/len(vals)) for i in range(len(vals))] 
    max_num=len(vals)
    width=1./(max_num+1.)
    x_ticks_pos=[j for j in range(len(vals[0]))]
    x_ticks_name=[r'$\mathrm{'+str(q[j])+'_{'+str(q_sz[j])+'}}$' for j in range(len(q))]
    
    plt.gca().invert_xaxis() 
    for i in range(len(vals[0])):
       for j in range(len(colors)):
            x=i+j*width-max_num/2.*width 
            plt.bar(x,vals[j,i],width=width,color=colors[j])
    plt.ylim(0,1) 
    plt.xticks(x_ticks_pos,x_ticks_name,rotation=90)
     
    plt.savefig('./Fits_details/'+title+'_meso_hel.pdf')
    hel_meso_all=np.array(vals) 
    hel_meso_mean=np.zeros((len(hel_meso_all[0])))
    hel_meso_std=np.zeros((len(hel_meso_all[0])))
    for i in range(len(hel_meso_all[0])):
        hel_meso_mean[i]=np.mean(hel_meso_all[:,i])
        hel_meso_std[i]=np.std(hel_meso_all[:,i])

    x=[i for i in range(len(hel_meso_mean))]
    x_ticks_pos=[j for j in range(len(hel_meso_mean))]
    x_ticks_name=[r'$\mathrm{'+str(q[j])+'_{'+str(q_sz[j])+'}}$' for j in range(len(q))]
    
    plt.close()
    colors=[cm.jet_r(float(i)/len(hel_meso_mean)) for i in range(len(hel_meso_mean))]

    plt.xlabel('Mesostate')
    plt.ylabel('Fractional helicity')
    plt.bar(x,hel_meso_mean,yerr=hel_meso_std,color=colors)
    plt.xticks(x_ticks_pos,x_ticks_name,rotation=90)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.ylim(0,1) 
    plt.savefig('./Fits_details/'+title+'_std_per_meso.pdf')
    W=''
    for i in range(len(hel_meso_mean)):
        W+=str(q[i])+'\t'+str(hel_meso_mean[i])+'\n'

    FU.write_file(title+'_std_per_meso.txt',W)

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps","-ns",help='Number of steps in the fitting',default=0)
    parser.add_argument("--num_restart","-nr",help='Number of restarts in the fitting',default=0)
    parser.add_argument("--plot_only","-po",help='',default=0)
    parser.add_argument("--top_fraction","-tp",help='',default=0.1)
    args = parser.parse_args() 
    if args.num_steps :
        try : 
            num_steps=int(args.num_steps)
        except : 
            print("Invalid ")
    else : 
        num_steps=100000

    if args.top_fraction :
        try : 
            top_frac=int(args.top_fraction)
        except : 
            print("Invalid ")
    else : 
        top_frac=0.1

    if args.plot_only :
        plot_only=False
        try : 
            if int(args.plot_only)==1:
                plot_only=True
        except : 
            print("Invalid ")
    else : 
        plot_only=False

    if args.num_restart :
        try : 
            num_restart=int(args.num_restart)
        except : 
            print("Invalid ")
    else : 
        num_restart=10

    use_raw=False #True#False
    linearize=False

    data=FU.read_file('folders.txt')
    loc_org=data[0][1]+'/'
    loc_hel=data[1][1]+'/'
    loc_dis=data[2][1]+'/'
    seq_3=QC.convert_residues_to_titrable(loc_org+'seq.in')
    seq_1=QC.convert_AA_3_to_1_letter(seq_3)

    CD_data=np.array(np.array(FU.read_file('CD_vs_pH.txt')),dtype=float) 
    CD_data_raw=np.array(np.array(FU.read_file('CD_vs_pH_raw.txt'))[1:],dtype=float)
    
    end_point_add=True
    if end_point_add==True :
        CD_data=np.array([[0,CD_data[0,1],CD_data[0,2]]]+np.ndarray.tolist(CD_data)+[[14,CD_data[-1,1],CD_data[-1,2]]])        
        CD_data_raw=np.array([[0,CD_data_raw[0,1],CD_data_raw[0,2]]]+np.ndarray.tolist(CD_data_raw)+[[14,CD_data_raw[-1,1],CD_data_raw[-1,2]]])

    if use_raw:
        CD_data_used=CD_data_raw
    else :
        CD_data_used=CD_data
    
    temp=FU.read_file('./Charge_layers.txt')
    q=np.array(np.array(temp)[:,0],dtype=int)
    q_sz=np.array(np.array(temp)[:,1],dtype=int)
    
    T=QC.read_param()[0] 
    top_frac=0.1
    pH=np.linspace(0,14,140) 
    data=FU.read_file('pKas_in.txt')
    pKas=[float(data[l][0]) for l in range(len(data))]
    DF=np.zeros((len(data)))    
    for u in range(len(pKas)):
        DF[u]=float(pKas[u])*R*T*np.log(10.)
    Fake_F=np.zeros((len(DF)+1))
    for i in range(len(DF)):
        Fake_F[i]=np.sum(DF[i:])

    F_new=np.array([[Fake_F[i]] for i in range(len(Fake_F))]) 
    Fs_err=F_new*0.

    Mapping=[]
    count=0
    for j in range(len(F_new)):
        Mapping+=[[]]
        for k in range(len(F_new[j])):
            Mapping[j]+=[count]
            count+=1

    n=len(seq_1)-1
    coil_signal=640-45*(T-273.15)
    fraction_hel_raw=(CD_data_raw[:,1]-coil_signal)/(-42500*(1-(3./n))-coil_signal)
    fraction_hel=CD_data[:,1]

    plt.scatter(CD_data[:,0],fraction_hel,color='b',label='CD deconvolution')
    plt.scatter(CD_data_raw[:,0],fraction_hel_raw,color='r',label='222nm based')
    plt.legend()
    plt.ylim(0.,1.)
    plt.savefig('./Fits_details/Fractional.png')
    plt.clf()

    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(F_new,Fs_err,pH,T,ign_norm_err=True,unsafe=0)
    QC.plot_probas(Proba,Proba_err,Mapping,pH)

    plt.clf()
    trans_pH=-(F_new[1:]-F_new[:-1])/(R*T*np.log(10.))

    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(F_new,Fs_err,pH,T,ign_norm_err=True,unsafe=0)
    
    if use_raw :
        fraction_hel_used=fraction_hel_raw
    else : 
        fraction_hel_used=fraction_hel

    pH=np.arange(0,14,0.1,dtype=np.float) 
    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(F_new,Fs_err,pH,T,ign_norm_err=True,unsafe=0)

    peak_pos=QC.get_proba_peak_pos(Proba)
    peak_pos=[len(pH)-1]+peak_pos

    Proba=np.array(Proba) 
    done=False
    count=0

    if linearize :
        data_new=[]
        pH_new=[]
        for p1 in range(len(CD_data_used[:,0])-1):
            for p2 in range(len(pH)):
                if pH[p2]>=CD_data_used[p1,0] and pH[p2]<=CD_data_used[p1+1,0]:
                    data_new+=[fraction_hel_used[p1]+(pH[p2]-CD_data_used[p1,0])*(fraction_hel_used[p1+1]-fraction_hel_used[p1])/(CD_data_used[p1+1,0]-CD_data_used[p1,0])]
                    pH_new+=[pH[p2]]
                elif pH[p2]>=CD_data_used[p1+1:0]:
                    break 
    else :
        data_new=fraction_hel_used
        pH_new=CD_data_used[:,0]
    plt.close()
    pH_new=np.array(pH_new)
    data_new=np.array(data_new) 
    plt.plot(pH_new,data_new)
    plt.scatter(CD_data_used[:,0],fraction_hel_used)
    plt.savefig('./Fits_details/test.png')

    if plot_only==False :
        hel_meso_all=get_target_per_meso(data_new,pH_new,F_new,T) 
        
    else :
        hel_meso_all=[]
        for i in range(num_restart):
            loc= 'Fits_details/Round_'+str(i)    
            temp=FU.read_file(loc+'/Hel_per_meso.txt')     
            hel_meso_all+=[[float(temp[i][1]) for i in range(len(temp))]]
 
    plt.close()
    
    for i in range(len(hel_meso_all)):
        hel_pH=np.zeros((len(pH)))
        loc= 'Fits_details/Round_'+str(i)
        for p in range(len(pH)):
            for s in range(len(hel_meso_all[i])):
                hel_pH[p]+=Proba[s,p]*hel_meso_all[i][s]
        plt.plot(pH,hel_pH,color='b')
    plt.scatter(CD_data_used[:,0],fraction_hel_used,color='orange')
    plt.xlabel('pH')
    plt.ylabel('Fractional helicity')
    plt.xlim(0,14)
    plt.ylim(0,1)
    plt.savefig('All_in_one.pdf')
    hel_pH=np.zeros((len(hel_meso_all),len(pH)))
    for i in range(len(hel_meso_all)): 
        loc= 'Fits_details/Round_'+str(i)        
        for p in range(len(pH)):
            for s in range(len(hel_meso_all[i])):
                hel_pH[i][p]+=Proba[s,p]*hel_meso_all[i][s]
    hel_mean=np.zeros((len(pH)))
    hel_std=np.zeros((len(pH)))
    for p in range(len(pH)):
        hel_mean[p]=np.mean(hel_pH[:,p])
        hel_std[p]=np.std(hel_pH[:,p])
    plt.close() 
    plt.fill_between(pH,hel_mean-hel_std,hel_mean+hel_std,alpha=0.5,color='g') 

    plt.scatter(CD_data_used[:,0],fraction_hel_used,color='orange')
    plt.xlabel('pH')
    plt.ylabel('Fractional Helicity')
    plt.ylim(0,1)
    plt.xlim(0,14)
    plt.savefig('./Fits_details/Fit_std_dev.pdf')
    plt.close()
    
    plt.scatter(CD_data_used[:,0],fraction_hel_used,color='orange')
    plt.xlabel('pH')
    plt.ylabel('Fractional Helicity')
    plt.ylim(0,1)
    plt.xlim(0,14)
    plt.savefig('./Fits_details/Fractional_helicity.pdf')
 
    Weights_norm,Weights_norm_err,Proba_new,Proba_new_err=QC.get_probas(F_new,F_new,pH_new,T,ign_norm_err=True,unsafe=0)
    ind_best=0
    Proba_new=np.array(Proba_new)

    E_fit_all=np.zeros((len(hel_meso_all)))
    E_fit_best=float('+inf')
    for i in range(len(hel_meso_all)):        
        E_fit=get_E_fit(Proba_new,hel_meso_all[i],data_new)
        print(i,E_fit)
        E_fit_all[i]=E_fit
        if E_fit_best>E_fit :
            E_fit_best=E_fit
            ind_best=i
    hel_meso_all=np.array(hel_meso_all) 
    max_ind=int(len(hel_meso_all)*top_frac)
    order_ind=np.argsort(E_fit_all)

    hel_meso_top=hel_meso_all[order_ind[:max_ind]]
    plot_hist_mix(hel_meso_top,'Top')
    plot_hist_mix(hel_meso_all,'All')
    plot_hist_mix([hel_meso_all[max_ind]],'Best') 
    hel_pH=np.zeros((len(pH)))
    loc='Fits_details/Round_'+str(i)    

    arr=np.array(hel_meso_all)
    avg=np.mean(hel_meso_all,axis=0)

    print_fig(avg,pH_new,fraction_hel_used,F_new,'./Fits_details/Average_meso')
    print('ind best ', ind_best)
    print_fig(hel_meso_all[ind_best],pH_new,fraction_hel_used,F_new,'./Fits_details/Best_')

    for i in range(len(hel_meso_top)):
        hel_pH=np.zeros((len(pH)))
        loc= 'Fits_details/Round_'+str(i)
        for p in range(len(pH)):
            for s in range(len(hel_meso_top[i])):
                hel_pH[p]+=Proba[s,p]*hel_meso_top[i][s]
        plt.plot(pH,hel_pH,color='b')
    plt.scatter(CD_data_used[:,0],fraction_hel_used,color='orange')
    plt.xlabel('pH')
    plt.ylabel('Fractional helicity')
    plt.xlim(0,14)
    plt.ylim(0,1)
    plt.savefig('./Fits_details/Top_'+str(top_frac)+'.pdf')

    RMSD_all=np.zeros((len(hel_meso_top)))

    for i in range(len(hel_meso_top)): 
        RMSD_all[i]=np.sqrt(np.sum((hel_meso_top[i]-np.mean(hel_meso_top[i]))**2))
    sorted_top=hel_meso_top[np.argsort(RMSD_all)]

    print_fig(sorted_top[0],pH_new,fraction_hel_used,F_new,'./Fits_details/Min_RMSD')
    plot_hist_mix([sorted_top[0]],'Min_RMSD')  
    plot_hist_mix([sorted_top[-1]],'Max_RMSD')
    W=''
    for i in range(len(hel_meso_all[ind_best])):
        W+=str(q[i])+'\t'+str(hel_meso_all[ind_best][i])+'\n'
    FU.write_file('./Fits_details/best_meso_hel.txt',W)
    
