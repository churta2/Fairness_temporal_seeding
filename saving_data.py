import pickle
import pandas as pd
import numpy as np
import statistics


z=0
table=np.zeros((15,5))
partial_table=np.zeros((5,5))



for l in range(5):
     
    
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/1k_data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_0+.pkl'
    temp_list=pd.read_pickle(file_name)
    partial_table[l,0]=0
    partial_table[l,1]=l+1
    partial_table[l,2]=(statistics.mean(temp_list))
    partial_table[l,3]=(statistics.mean(temp_list)- 1.96*(statistics.pstdev(temp_list)/np.sqrt(30)))
    partial_table[l,4]=(statistics.mean(temp_list)+ 1.96*(statistics.pstdev(temp_list)/np.sqrt(30)))
    

table[0:5,:]=partial_table
        
z=0.5
partial_table=np.zeros((5,5))


for l in range(5):
     
    
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/1k_data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_0+.pkl'
    temp_list=pd.read_pickle(file_name)
    partial_table[l,0]=0.5
    partial_table[l,1]=l+1
    partial_table[l,2]=(statistics.mean(temp_list))
    partial_table[l,3]=(statistics.mean(temp_list)- 1.96*(statistics.pstdev(temp_list)/np.sqrt(30)))
    partial_table[l,4]=(statistics.mean(temp_list)+ 1.96*(statistics.pstdev(temp_list)/np.sqrt(30)))
    
table[5:10,:]=partial_table
        

z=1
partial_table=np.zeros((5,5))


for l in range(5):
     
    
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/1k_data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_0+.pkl'
    temp_list=pd.read_pickle(file_name)
    partial_table[l,0]=1
    partial_table[l,1]=l+1
    partial_table[l,2]=(statistics.mean(temp_list)) 
    partial_table[l,3]=(statistics.mean(temp_list)- 1.96*(statistics.pstdev(temp_list)/np.sqrt(30))) 
    partial_table[l,4]=(statistics.mean(temp_list)+ 1.96*(statistics.pstdev(temp_list)/np.sqrt(30)))
    

table[10:15,:]=partial_table
        






z=0
table2=np.zeros((15,5))
partial_table=np.zeros((5,5))

for l in range(5):
     
    
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/previos_Data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_1+.pkl'
    temp_list=pd.read_pickle(file_name)
    partial_table[l,0]=0
    partial_table[l,1]=l+1
    partial_table[l,2]=statistics.mean(temp_list)
    partial_table[l,3]=statistics.mean(temp_list)- 1.96*(statistics.pstdev(temp_list)/np.sqrt(30))
    partial_table[l,4]=statistics.mean(temp_list)+ 1.96*(statistics.pstdev(temp_list)/np.sqrt(30))
    
    

table2[0:5,:]=partial_table
        
z=0.5
partial_table=np.zeros((5,5))

for l in range(5):
     
    
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/previos_Data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_1+.pkl'
    temp_list=pd.read_pickle(file_name)
    partial_table[l,0]=0.5
    partial_table[l,1]=l+1
    partial_table[l,2]=statistics.mean(temp_list)
    partial_table[l,3]=statistics.mean(temp_list)- 1.96*(statistics.pstdev(temp_list)/np.sqrt(30))
    partial_table[l,4]=statistics.mean(temp_list)+ 1.96*(statistics.pstdev(temp_list)/np.sqrt(30))
    
    

table2[5:10,:]=partial_table
        

z=1
partial_table=np.zeros((5,5))

for l in range(5):
     
    
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/previos_Data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_1+.pkl'
    temp_list=pd.read_pickle(file_name)
    partial_table[l,0]=1
    partial_table[l,1]=l+1
    partial_table[l,2]=statistics.mean(temp_list)
    partial_table[l,3]=statistics.mean(temp_list)- 1.96*(statistics.pstdev(temp_list)/np.sqrt(30))
    partial_table[l,4]=statistics.mean(temp_list)+ 1.96*(statistics.pstdev(temp_list)/np.sqrt(30))
    
    

table2[10:15,:]=partial_table
        



table3=np.zeros((5,4))


table3[:,0:2]=table[0:5,0:2]
table3[:,2]=table[0:5,3]
table3[:,3]=table2[0:5,3]


table4=np.zeros((5,4))


table4[:,0:2]=table[5:10,0:2]
table4[:,2]=table[5:10,3]
table4[:,3]=table2[5:10,3]


table5=np.zeros((5,4))


table5[:,0:2]=table[10:15,0:2]
table5[:,2]=table[10:15,3]
table5[:,3]=table2[10:15,3]

table6=np.concatenate([table3, table4, table5],axis=0)



df = pd.DataFrame(table)


file_name_df='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/1k_data/zero_correlation.csv'


df.to_csv(file_name_df,index=False)
