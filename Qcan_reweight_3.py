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
import shutil

##!!!!!!!!!!!!!! WARNING : this code is an absolute mess !!!!!!!!!!!!!!! 
# Getting better 
R=1.987*10**(-3)
plt.rcParams["font.family"]="Times New Roman"
def get_meso_AH(AH,counts):
    Meso_AH=np.zeros((len(counts)))
    for m in range(len(counts)):
        for n in range(len(counts[m])):
            Meso_AH[m]+=counts[m][n]*np.mean(AH[m][n])
    return Meso_AH

def get_mixed_quantity(val_dis,val_org,val_hel,mix,q_offset,layers_q):
    val_out=[]
    for i in range(len(val_dis)):
        if not (layers_q[i]>=q_limits[0] and layers_q[i]<=q_limits[1]):
            val_out+=[[float('nan')]]
            continue
             
        val_out+=[[]]
        for j in range(len(val_dis[i])):
            val_out[i]+=[np.array(val_dis[i])[j]*mix[i+q_offset,0]+np.array(val_org[i])[j]*mix[i+q_offset,1]+np.array(val_hel[i])[j]*mix[i+q_offset,2]]
    return val_out

def get_mixed_quantity_per_res(val_dis,val_org,val_hel,mix,q_offset,layers_q):
    val_out=[]
    for i in range(len(AH_org)):
        if not (layers_q[i]>=q_limits[0] and layers_q[i]<=q_limits[1]):
            val_out+=[[float('nan')]]
            continue

        val_out+=[[]]
        for j in range(len(AH_org[i])):
            val_out[i]+=[[]]
            for k in range(len(AH_org[i][j])) :   
                val_out[i][j]+=[mix[i+q_offset,0]*val_dis[i][j][k]+mix[i+q_offset,1]*val_org[i][j][k]+mix[i+q_offset,2]*val_hel[i][j][k]]
    return val_out

def plot_hist_mix(vals,title):
    plt.close()
    vals=np.array(vals) 
    colors=[cm.jet_r(float(i)/len(vals[0])) for i in range(len(vals[0]))] 
    max_num=len(vals[0])
    width=1./(max_num+1.)
    labels=['dis','org','hel']
    x_ticks_pos=[j for j in range(len(vals))]
    x_ticks_name=[r'$\mathrm{'+str(q[j])+'_{'+str(q_sz[j])+'}}$' for j in range(len(q))]
    for i in range(len(vals)):
        for j in range(len(colors)):
            x=i+j*width-max_num/2.*width 
            plt.bar(x,vals[i,j],width=width,color=colors[j],label=labels[j])
    plt.legend() 
    plt.xticks(x_ticks_pos,x_ticks_name) 
    plt.savefig('./'+title+'.pdf')
    plt.close()
def auto_get_mixed_quantity(mix,folders,name):
    val_dis=QC.read_HSQ_format(folders[0]+'Results/'+name,False)
    val_org=QC.read_HSQ_format(folders[1]+'Results/'+name,False)
    val_hel=QC.read_HSQ_format(folders[2]+'Results/'+name,False)
    
    val_out=get_mixed_quantity(val_dis,val_org,val_hel,mix,q_offset,layers_q)
    QC.write_HSQ_format(val_out,'Results/'+name,False)

def auto_get_mixed_quantity_per_res(mix,folders,name,layers_q):

    val_dis=QC.read_HSQ_format(folders[0]+'Results/'+name,True)    
    val_org=QC.read_HSQ_format(folders[1]+'Results/'+name,True)
    val_hel=QC.read_HSQ_format(folders[2]+'Results/'+name,True)
    
    val_out=get_mixed_quantity_per_res(val_dis,val_org,val_hel,mix,q_offset,layers_q)

    QC.write_HSQ_format(val_out,'Results/'+name,True)

def get_meso_DF(F_meso):                                                                 
    DF=[]
    for i in range(len(F_meso)):
        DF+=[[]]
        for j in range(len(F_meso[i])):
            DF[i]+=[F_meso[i][j]-min(F_meso[i])]
        DF[i]=np.array(DF[i])
    return DF

if __name__=="__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps","-ns",help='Number of steps in the fitting',default=0)
    parser.add_argument("--num_restart","-nr",help='Number of restarts in the fitting',default=0)    
    parser.add_argument("--q_limits","-q",help='Limits of the simulation used',nargs='+')
    args = parser.parse_args() 

    if args.q_limits :
        if len(args.q_limits)!=2 :
            print("The q_limits argument must have two entries")
        else : 
            q_limits=[]
            for i in range(2):
                try :                                                                                                 
                    q_limits+=[int(args.q_limits[i])]
                except :
                    print("Only integer values are valid for q_limits")
    else :
        q_limits=[]

    use_raw=False
    linearize=False

    data=FU.read_file('folders.txt')
    loc_org=data[0][1]+'/'
    loc_hel=data[1][1]+'/'
    loc_dis=data[2][1]+'/'
    seq_3=QC.convert_residues_to_titrable(loc_org+'seq.in')
    seq_1=QC.convert_AA_3_to_1_letter(seq_3)
    
    target_hel_meso=FU.read_file('./Fits_details/best_meso_hel.txt')

    CD_data=np.array(np.array(FU.read_file('CD_vs_pH.txt')),dtype=float)        
    CD_data_raw=np.array(np.array(FU.read_file('CD_vs_pH_raw.txt'))[1:],dtype=float)

    if use_raw:
        CD_data_used=CD_data_raw
    else :
        CD_data_used=CD_data

    temp=FU.read_file(loc_org+'Charge_layers.txt')
    q_org=np.array(np.array(temp)[:,0],dtype=int)
    shutil.copyfile(loc_org+'Charge_layers.txt','./Charge_layers.txt')
    temp=FU.read_file('./Charge_layers.txt') 
    q=np.array(np.array(temp)[:,0],dtype=int)
    if len(target_hel_meso)!= len(q):
        print(len(target_hel_meso),len(q))
        print("The charge layer file must correspond to the target mesostates helicity used.")
        quit()
    
    layers_q=np.array(np.array(temp)[:,0],dtype=int)
    q_sz=np.array(np.array(temp)[:,1],dtype=int)

    q_offset=q_org[0]-q[0]
 
    T=QC.read_param()[0] 
   
    pH=np.linspace(1,14,140) 
    data=FU.read_file('pKas_in.txt')
    pKas=[float(data[l][0]) for l in range(len(data))]
    DF=np.zeros((len(data)))
    for u in range(len(pKas)):
        DF[u]=float(pKas[u])*R*T*np.log(10.)
    
    Fake_F=np.zeros((len(DF)+1))
    for i in range(len(DF)):
        Fake_F[i]=np.sum(DF[i:])

    F_new=np.array([[Fake_F[i]] for i in range(len(Fake_F))]) 

    W=''
    for i in range(len(q)):
        W+=str(F_new[i][0])+'\t'+str(0.)+'\n'

    FU.write_file('./Results/Fs/Global_F_per_level.txt',W)
    
    Fs_err=F_new*0.

    Mapping=[]
    count=0
    for j in range(len(F_new)):
        Mapping+=[[]]
        for k in range(len(F_new[j])):
            Mapping[j]+=[count]
            count+=1

    n=len(F_new)+4

    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(F_new,Fs_err,pH,T,ign_norm_err=True,unsafe=0)


    plt.clf()
    trans_pH=-(F_new[1:]-F_new[:-1])/(R*T*np.log(10.))
    #Not sure what those are for
    Fs_org=QC.read_HSQ_format(loc_org+'Results/Fs/Normal/F_per_state.txt',False)
    Fs_hel=QC.read_HSQ_format(loc_hel+'Results/Fs/Normal/F_per_state.txt',False)
    Fs_dis=QC.read_HSQ_format(loc_dis+'Results/Fs/Normal/F_per_state.txt',False)    
    # So here we will adapt the innner probabilities of each mesosattes such that the differene is maintained within, and that the overall F is the one specified
    for i in range(len(Fs_org)):
        Fs_org[i]=QC.adapt_microstates_weight(Fs_org[i],Fake_F[i],T)
        Fs_hel[i]=QC.adapt_microstates_weight(Fs_hel[i],Fake_F[i],T)
        Fs_dis[i]=QC.adapt_microstates_weight(Fs_dis[i],Fake_F[i],T)
    
    AH_org=QC.read_HSQ_format(loc_org+'Results/Plots/Values_to_plot/Alpha_helix.txt',True)
    AH_hel=QC.read_HSQ_format(loc_hel+'Results/Plots/Values_to_plot/Alpha_helix.txt',True)
    AH_dis=QC.read_HSQ_format(loc_dis+'Results/Plots/Values_to_plot/Alpha_helix.txt',True) 

    plt.close()

    counts_org=QC.read_HSQ_format(loc_org+'Results/States_details/Population_counts.txt',False)
    counts_hel=QC.read_HSQ_format(loc_hel+'Results/States_details/Population_counts.txt',False)
    counts_dis=QC.read_HSQ_format(loc_dis+'Results/States_details/Population_counts.txt',False)
 
    FU.check_and_create_rep('./Results/States_details')
    #shutil.copyfile(loc_org+'/Results/States_details/States_mapping.txt','./Results/States_details/States_mapping.txt') 
    #shutil.copyfile(loc_org+'/Results/States_details/States.txt','./Results/States_details/States.txt')


    org_M_AH=get_meso_AH(AH_org,counts_org)
    dis_M_AH=get_meso_AH(AH_dis,counts_dis)
    hel_M_AH=get_meso_AH(AH_hel,counts_hel)
    stp_sz=0.001   
    mix=np.array([[0.,1.,0.] for i in range(len(target_hel_meso))])

    # I think insteead of working with helicity per mesotates, we also need to account for what exactly is the difference in overall helicity change for each microstates when you go from one ensemble to the other. 
    # Unfortunatly, that means even the weight need to be considered "live"
    # A little trickier to code, but worth it

    for i in range(len(org_M_AH)):        
        if not (layers_q[i]>=q_limits[0] and layers_q[i]<=q_limits[1]):
            continue
        target=float(target_hel_meso[i+q_offset][1])
        while True :
            mixed=mix[i+q_offset,0]*dis_M_AH[i]+mix[i+q_offset,1]*org_M_AH[i]+mix[i+q_offset,2]*hel_M_AH[i] 
            if abs(mixed-target)<stp_sz:
                break
            elif mixed>target:
                mix[i+q_offset,1]+=-stp_sz
                mix[i+q_offset,0]+=stp_sz
                if mix[i+q_offset,0]>=1:
                    break
            elif mixed<target:
                mix[i+q_offset,1]+=-stp_sz
                mix[i+q_offset,2]+=stp_sz
                if mix[i+q_offset,2]>=1:
                    break
    for i in range(len(org_M_AH)):
        print(layers_q[i],mix[i],target_hel_meso[i],mix[i,0]*dis_M_AH[i]+mix[i,1]*org_M_AH[i]+mix[i,2]*hel_M_AH[i])

    plot_hist_mix(mix,'mix_per_meso')
    pH=np.arange(0,14,0.1,dtype=np.float) 
    Weights_norm,Weights_norm_err,Proba,Proba_err=QC.get_probas(F_new,Fs_err,pH,T,ign_norm_err=True,unsafe=0)
    
    DFs_org=get_meso_DF(Fs_org)
    DFs_dis=get_meso_DF(Fs_dis)
    DFs_hel=get_meso_DF(Fs_hel)
 
    mixed_DF=get_mixed_quantity(DFs_dis,DFs_org,DFs_hel,mix,q_offset,layers_q)

    Fs_rew_rev=[]
    # Now get the difference to have a mean value of zero, and add it to the Fake_F 
    for i in range(len(mixed_DF)):        
        full=-R*T*np.log(np.sum([np.exp(-mixed_DF[i][j]/(R*T))for j in range(len(mixed_DF[i]))]))

        Fs_rew_rev+=[[Fake_F[i+q_offset]+mixed_DF[i][j]-full for j in range(len(mixed_DF[i]))]]

    QC.write_HSQ_format(Fs_rew_rev,'./Results/Fs/Invert/F_per_state.txt',False)     

    sub=np.amax(Fs_rew_rev)
    Fs_rew_nor=[]
    Fs_rew_err=[] # We just need an error for the plotting, we will set it to zero 
    for i in range(len(Fs_rew_rev)):
        Fs_rew_nor+=[[Fs_rew_rev[i][j]-sub for j in range(len(Fs_rew_rev[i]))]]
        Fs_rew_err+=[[0. for j in range(len(Fs_rew_rev[i]))]]
 
    QC.write_HSQ_format(Fs_rew_nor,'./Results/Fs/Invert/F_per_state_err.txt',False)
    QC.write_HSQ_format(Fs_rew_nor,'./Results/Fs/Normal/F_per_state.txt',False)
    QC.write_HSQ_format(Fs_rew_nor,'./Results/Fs/Normal/F_per_state_err.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Normal/S_per_state.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Normal/S_per_state_err.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Normal/U_per_state.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Normal/U_per_state_err.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Invert/S_per_state.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Invert/S_per_state_err.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Invert/U_per_state.txt',False)
    QC.write_HSQ_format(Fs_rew_err,'./Results/Fs/Invert/U_per_state_err.txt',False)
    
    AH_out=get_mixed_quantity_per_res(AH_dis,AH_org,AH_hel,mix,q_offset,layers_q)
    plt.ylim(0,1.) 
    val_err=0
     
    folders=[loc_dis,loc_org,loc_hel]

    colors=['red','green','orange']
    plt.clf()
    x=[-i for i in range(len(AH_out))]

    x_ticks_pos=[-j for j in range(len(mix))]
    x_ticks_name=[r'$\mathrm{'+str(q[j])+'_{'+str(q_sz[j])+'}}$' for j in range(len(q))] 

    get_meso_AH(AH_hel,counts_hel)
    for i in range(len(AH_out)):
        plt.bar(-i,np.mean(AH_out[i]),color='b')

    plt.xticks(x_ticks_pos,x_ticks_name)
    plt.savefig('Helicity.pdf')

    vals=mix
    
    names=[ 'Fs/Normal/S_per_state.txt',\
            'Fs/Normal/S_per_state_err.txt',\
            'Fs/Invert/S_per_state.txt',\
            'Fs/Invert/S_per_state_err.txt',\
            'Fs/Normal/U_per_state.txt',\
            'Fs/Normal/U_per_state_err.txt',\
            'Fs/Invert/U_per_state.txt',\
            'Fs/Invert/U_per_state_err.txt',\
            'Fs/Normal/F_per_state.txt',\
            'Fs/Normal/F_per_state_err.txt',\
            'Fs/Invert/F_per_state.txt',\
            'Fs/Invert/F_per_state_err.txt',\
            'Plots/Values_to_plot/Dipolar_moment.txt',\
            'Plots/Values_to_plot/SLT_BR.txt',
            'Plots/Values_to_plot/HET_CONF.txt'] 

    names_per_res=['Plots/Values_to_plot/Alpha_helix.txt','Plots/Values_to_plot/Alpha_helix_err.txt','Plots/Values_to_plot/Beta_sheet.txt','Plots/Values_to_plot/Beta_sheet_err.txt','Plots/Values_to_plot/Random_coil.txt','Plots/Values_to_plot/Random_coil_err.txt','Plots/Values_to_plot/CS_HA.txt','Plots/Values_to_plot/Hel_length.txt']
    for i in range(len(names_per_res)):
        try : 
            auto_get_mixed_quantity_per_res(mix,folders,names_per_res[i],layers_q)
        except : 
            print("Could not interpolate "+names_per_res[i])
    for i in range(len(names)):
        try :  
            auto_get_mixed_quantity(mix,folders,names[i])
        except : 
            print("Could not interpolate "+names[i]) 
    pops=[]
    for i in range(len(layers_q)):
        if not (layers_q[i]>=q_limits[0] and layers_q[i]<=q_limits[1]):
            pops+=[[float('nan')]]
            continue 
        Fs_rew_nor[i]=Fs_rew_rev[i] 
        pops+=[QC.get_microstates_population_mesostates(Fs_rew_nor[i],Fs_rew_rev[i],T)[0]]
    QC.plot_probas(Proba,Proba_err,pops,pH)
    QC.write_HSQ_format(pops,'./Results/States_details/Population_counts.txt', False)
    QC.write_HSQ_format(pops,'./Results/States_details/Population_counts_err.txt', False)
    
