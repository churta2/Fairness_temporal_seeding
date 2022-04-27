import dask
from dask import delayed
import numpy as np
import pandas as pd
import networkx as nx
import random
import pickle
# import matplotlib.pyplot as plt
import copy
import statistics
# conda install graphviz
#import graphviz
import os
from scipy.stats import pearsonr


class BaseGraphs:
    
    def __init__(self,G):
        self.G=G
        
    def add_endowments(self):
        df=pd.DataFrame(list(np.random.beta(a=1, b=1, size=len(self.G.nodes()))))
        endowment=df.to_dict()
        nx.set_node_attributes(self.G,endowment[0], name="w" )
        
        
    def add_endowments_prior(self, list_endw):
        
        df=pd.DataFrame(list_endw)
        endowment=df.to_dict()
        nx.set_node_attributes(self.G,endowment[0], name="w" )
        
        
        
    def add_endowments_perfect_correlated(self):
        
        norm=max([self.G.degree(i) for i in self.G.nodes()])
        df=pd.DataFrame([self.G.degree(i)/norm for i in self.G.nodes()])
        endowment=df.to_dict()
        nx.set_node_attributes(self.G,endowment[0], name="w" )
        
        
    
    def add_endowments_partial_correlated(self):
        
        norm=max([G.degree(i) for i in G.nodes()])
        degrees=[G.degree(i)/norm for i in G.nodes()]
        
        noise=[ np.random.normal(i, 0.2, 1)[0] for i in degrees]
        adj_below=[0.03 if i<0 else i for i in noise]
        adj_abov=[1 if i>1 else i for i in adj_below]

        df=pd.DataFrame(adj_abov)
        endowment=df.to_dict()
        nx.set_node_attributes(self.G,endowment[0], name="w" )


        
    
    def add_time_delays(self):
                
        df=pd.DataFrame(list(np.random.uniform(low=0, high=5, size=len(self.G.nodes()))))
        times=df.to_dict()
        nx.set_node_attributes(self.G,times[0], name="t" )
        
    

        
    ### n is the number of edges not nodes
    def live_edges_saving(self,n,p,step,sample_size):
        k=int(n*p)
        sampled_edges = random.sample(self.G.edges, k)
        
        ini_nodes=set(G.nodes())
                
        sG=self.G.edge_subgraph(sampled_edges).copy()
        
        s_nodes=set(sG.nodes())
        
        diff_nodes=list(ini_nodes.difference(s_nodes))
        
        for i in diff_nodes:
            sG.add_node(i, w= G.nodes[i]['w'], t=G.nodes[i]['t'] )

        
        # path='Live_edges\live_edges'+ str(step)+'_'+str(sample_size)+'.pkl'
        # pickle.dump(sG, open(path, 'wb'))
        
        return sG
    
    def live_edges(self,n,p):
        k=int(n*p)
        sampled_edges = random.sample(self.G.edges, k)
        ini_nodes=set(self.G.nodes())      
        sG=self.G.edge_subgraph(sampled_edges).copy()
        s_nodes=set(sG.nodes())
        diff_nodes=list(ini_nodes.difference(s_nodes))
        
        for i in diff_nodes:
            sG.add_node(i, w= G.nodes[i]['w'], t=G.nodes[i]['t'] )

        
        # path='Live_edges\live_edges'+ str(step)+'_'+str(sample_size)+'.pkl'
        # pickle.dump(sG, open(path, 'wb'))
        
        return sG

    def base_welfare(self):

        temp = [self.G.nodes[i]['w'] for i in self.G.nodes()]

        return temp


class Greedy_algorithm_1:
    
    
    def __init__(self,rho, tau,delta,k,alpha):
        self.rho=rho
        self.seed_set=set()
        self.tau=tau
        self.delta=delta
        self.welfare=0
        self.k=k
        self.alpha=alpha
        self.epsilon=0
    
    
    def epsilon_calculation(self):
        
        self.epsilon=(1-1/np.exp(1))*(self.delta**(self.tau/2))
        
        return (1-1/np.exp(1))*(self.delta**(self.tau/2))
    
    
    def update_sample_size(self,N):
        
        num= 9*self.k*np.log(6*N*self.k/self.epsilon)
        
        self.rho=int(np.ceil(num/self.epsilon))
        
        return int(np.ceil(num/self.epsilon))
        
  
    def query(self,G,nu):
        
        
        G2=copy.deepcopy(G)
        G2=G2.reverse()
        
        tree = nx.bfs_tree(G2, nu, depth_limit=10)
        # nx.draw(tree, with_labels=True)
        
        
        at1=G2.edge_subgraph(tree.edges).copy()
        
        at2 = copy.deepcopy(at1) 
        # nx.draw(at2, with_labels=True)



        leafs=[i for i in at1.nodes() if at1.degree(i)==1]

        for i in leafs:

            for path in nx.all_simple_paths(at1, source=nu, target=i):
                
                path=path[1:]
                #print('a possible path:' +str(path))
                time_delays=[at1.nodes[i]['t'] for i in path]
                #print('times:' +str(time_delays))
                temp=0
                indice=0
                for j in time_delays:
                
                    temp+=j
                    #print('vaaa' +str(temp))
                    if (temp>self.tau):
                       
                        s_path=path[indice:]
                        for k in s_path:
                            try:
                                at2.remove_node(k)
                            except:
                                pass
                        temp=0
                        break
                    
                    indice+=1
        
        output=at2.reverse()
        
        return output
        
        
     

    
    ###seed set is different from self.seed_set
    
    
    def welfare_evaluation(self, time_reverse_cascade, seed_set, sample_node):
        
           
        increment=0
        
        for s in list(seed_set):
            
            paths_to=[]
            
            try:
                for path in nx.all_simple_paths(time_reverse_cascade, source=s, target=sample_node):
                    
                    dist=0
                    for k in path[:-1]:
                        dist+= time_reverse_cascade.nodes[k]['t']
                        
                    paths_to.append(dist)    
                    
    
                if paths_to==[]:
                    paths_to.append(10000)
                
                
                min_dist=min(paths_to)
                increment+=self.delta**(min_dist)
                 
            except:
                
                increment+=0
                
        if self.alpha==0:
            
            obj=np.log((increment + time_reverse_cascade.nodes[sample_node]['w']))
            
        else:
            
            obj=((increment + time_reverse_cascade.nodes[sample_node]['w'])**self.alpha)/self.alpha
            
        return obj
    
   
    def next_seed(self,G, sample_nodes):
        
        ## generation of time reverse cascades 
        t=BaseGraphs(G)
        t.add_time_delays()
        #live_edges_j=[]
        list_Cj=[]
        
  
        ##comprehension list!!!!
       
        for i in sample_nodes:
            # print('Evaluating influence of node: ' +str(i))
            temp_live_edges=t.live_edges(len(G.edges()),0.5)
            #live_edges_j.append(temp_live_edges)
            list_Cj.append(self.query(temp_live_edges, i))
          
    

        # print('lista  :'+str(list_u))
        table=np.zeros((len(G.nodes),len(sample_nodes)))
        
        
        for j in range(len(sample_nodes)):
            
            t1=set(list_Cj[j].nodes())
            t2=t1.difference(self.seed_set)
            useful=t2.difference(set(sample_nodes))
            
            all_nodes=set(G.nodes())
            all_nodes_list=list(all_nodes)
            to_zero=all_nodes.difference(useful)  
            
  
            table[list(to_zero),j]=[0]*len(to_zero)
            
     
            for i in list(useful):
                
                set_bef= copy.deepcopy(self.seed_set) 
                set_aft=set_bef.union({i})
                

                table[i,j]= (self.welfare_evaluation( list_Cj[j], set_aft, sample_nodes[j]))- (self.welfare_evaluation( list_Cj[j], set_bef, sample_nodes[j]))
                
                
            
       
        row_totals = list(table.sum(axis=1))
       
        where=row_totals.index(max(row_totals))
        
        # print('============' +str(table))
        
        self.seed_set=self.seed_set.union({all_nodes_list[where]})
        
        #return list_u[where]
        return max(row_totals)/len(sample_nodes)
    
    def base_welfare(self,G):
        
        temp=[G.nodes[i]['w'] for i in G.nodes()]

        
        return temp
    
    
    def seeding(self,G,fake_input):
        
        print('simulation '+ str(fake_input))
        #table=[self.welfare]

        for i in range(self.k):
            
            print('Defining seed: ' +str(i))
            sample_nodes=random.sample(G.nodes(), self.rho)
            self.welfare+=self.next_seed(G, sample_nodes)
            #table.append(self.welfare)
        

        return self.seed_set


###1.  input the graph, initializing with the endowments



# G = nx.read_adjlist("higgs-reply_network.edgelist.gz")


def graph_reduction(G):
    
    K=nx.DiGraph()
    K.add_edges_from(list(G.edges()))

    
    all_edges= list(K.edges())
    sub_edges=random.sample(all_edges, 1000)

    H = K.edge_subgraph(sub_edges).copy()
    
    return H


def formating_graph(G):
    
    dim1=len(G.nodes())
    dim2=2
    table1=np.zeros((dim1,dim2))
    table1[:,0]=list(range(len(G.nodes())))
    table1[:,1]=[str(i) for i in G.nodes()]
    df = pd.DataFrame(table1)
    df['ind']=[str(i) for i in G.nodes()]
    df=df.set_index('ind')
    del df[1]
    df[0] = df[0].astype(np.int64)
    dict1=df.to_dict()

    J = nx.relabel_nodes(G, dict1[0])
    
    F = nx.DiGraph()
    F.add_edges_from(list(J.edges()))


    
    return F



def adding_links(G,m):
    

    
    L=copy.deepcopy(G)
    
    which_out_degree=[]
    
    for i in list(L.nodes()):
        
        if L.out_degree(i)==0:
            which_out_degree.append(i)
            
    
    which_in_degree=[]
    
    for i in list(L.nodes()):
        
        if L.in_degree(i)==0:
            which_in_degree.append(i)
       
          
       
    sample_in=random.sample(which_in_degree, m)
    sample_out=random.sample(which_out_degree, m)
        
    new_edges=[]
    
    for i in range(m):
          new_edges.append((sample_out[i],sample_in[i]))  
        
        
    L.add_edges_from(new_edges)  
    
    return L 



def adding_links_middle(G,m):
    
   
    L=copy.deepcopy(G)
    
    which_out_degree=[]
    
    for i in list(L.nodes()):
        
        if (L.out_degree(i)==1 & L.in_degree(i)==1):
            which_out_degree.append(i)
            
    
    which_in_degree=[]
    
    for i in list(L.nodes()):
        
        if (L.in_degree(i)==0):
            which_in_degree.append(i)
       
          
       
    sample_in=random.sample(which_in_degree, m)
    sample_out=random.sample(which_out_degree, m)
        
    new_edges=[]
    
    for i in range(m):
          new_edges.append((sample_out[i],sample_in[i]))  
        
        
    L.add_edges_from(new_edges)  
    
    return L 



def complex_graph(G):
    

    G=graph_reduction(G)
    G=formating_graph(G)
    
    
    G=adding_links(G,50)
    G=adding_links(G,50)
    G=adding_links_middle(G,30)
    G=adding_links_middle(G,50)
    G=adding_links(G,50)
    G=adding_links(G,50)
    G=adding_links_middle(G,50)
    G=adding_links(G,80)
    G=adding_links(G,50)
    
    return G


graph_name= 'C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/Final_graph.pkl'

# with open(graph_name, 'wb') as f:
#     pickle.dump(G, f)


with open(graph_name, 'rb') as f:
    G = pickle.load(f)

H=BaseGraphs(G)

H.add_endowments()

base_0=H.base_welfare()

with open('base0.pkl', 'wb') as f:
    pickle.dump(base_0, f)

### 2. Set the parameter of the greedy algorithm  (rho, tau,delta,k)
### rho is the number of the sample, tau the time limit, delta discount factor and k the number of seeds


alfas=[0,0.5,1]

for z in alfas:
 
    for l in range(5):
        
        file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_0+.pkl'



        out_par=[]
        
        simulation=list(range(30))
        
                
        for i in simulation:
            # (rho, tau,delta,k,alpha)
            q=Greedy_algorithm_1(50,l+1,0.8,10,z)

            #q.epsilon_calculation()
            #q.update_sample_size(len(H.G.nodes()))
        
        
            inter1=delayed(q.seeding)(H.G,i)
            out_par.append(inter1)
            
        
            
        salida= dask.compute(out_par)[0]
        
        
        print(salida)
        
        with open(file_name, 'wb') as f:
            pickle.dump(salida, f)

######################################################################################################################

graph_name = 'C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/Final_graph.pkl'

# with open(graph_name, 'wb') as f:
#     pickle.dump(G, f)

with open(graph_name, 'rb') as f:
    G = pickle.load(f)

H = BaseGraphs(G)

H.add_endowments_partial_correlated()

base_2=H.base_welfare()

with open('base2.pkl', 'wb') as f:
    pickle.dump(base_2, f)


### 2. Set the parameter of the greedy algorithm  (rho, tau,delta,k)
### rho is the number of the sample, tau the time limit, delta discount factor and k the number of seeds


alfas = [0, 0.5, 1]

for z in alfas:

    for l in range(5):

        file_name = 'C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/reply_alpha_' + str(
            z) + '_t_.' + str(l + 1) + '_rho_2+.pkl'

        out_par = []

        simulation = list(range(30))

        for i in simulation:
            # (rho, tau,delta,k,alpha)
            q = Greedy_algorithm_1(50, l + 1, 0.8, 10, z)
            # q.epsilon_calculation()
            # q.update_sample_size(len(H.G.nodes()))

            inter1 = delayed(q.seeding)(H.G, i)
            out_par.append(inter1)

        salida = dask.compute(out_par)[0]

        print(salida)

        with open(file_name, 'wb') as f:
            pickle.dump(salida, f)

graph_name = 'C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/Final_graph.pkl'

# with open(graph_name, 'wb') as f:
#     pickle.dump(G, f)

######################################################################################################################
with open(graph_name, 'rb') as f:
    G = pickle.load(f)

H = BaseGraphs(G)

H.add_endowments_perfect_correlated()

base_1=H.base_welfare()

with open('base1.pkl', 'wb') as f:
    pickle.dump(base_1, f)


### 2. Set the parameter of the greedy algorithm  (rho, tau,delta,k)
### rho is the number of the sample, tau the time limit, delta discount factor and k the number of seeds


alfas = [0, 0.5, 1]

for z in alfas:

    for l in range(5):

        file_name = 'C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/reply_alpha_' + str(
            z) + '_t_.' + str(l + 1) + '_rho_1+.pkl'

        out_par = []

        simulation = list(range(30))

        for i in simulation:
            # (rho, tau,delta,k,alpha)
            q = Greedy_algorithm_1(50, l + 1, 0.8, 10, z)
            # q.epsilon_calculation()
            # q.update_sample_size(len(H.G.nodes()))

            inter1 = delayed(q.seeding)(H.G, i)
            out_par.append(inter1)

        salida = dask.compute(out_par)[0]

        print(salida)

        with open(file_name, 'wb') as f:
            pickle.dump(salida, f)


graph_name = 'C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/Final_graph.pkl'

# with open(graph_name, 'wb') as f:
#     pickle.dump(G, f)

######################################################################################################################







with open(graph_name, 'rb') as f:
    G = pickle.load(f)

H = BaseGraphs(G)

list_endw_0=pd.read_pickle('./1k_data/base0.pkl')

H.add_endowments_prior(list_endw_0)
H.add_time_delays()

low_endw=[i for i in H.G.nodes() if H.G.nodes[i]['w'] < 0.25 ]

high_endw=[i for i in H.G.nodes() if H.G.nodes[i]['w'] > 0.75 ]



#welfare_evaluation(self, time_reverse_cascade, seed_set, sample_node).

cum_low_endw=sum([H.G.nodes[i]['w'] for i in low_endw])
cum_high_endw=sum([H.G.nodes[i]['w'] for i in high_endw])





 # (rho, tau,delta,k, alpha)



table=np.zeros((15,5))

z=0
partial_table=np.zeros((5,5))

q=Greedy_algorithm_1(50,1,0.8,10,z)

for l in range(5):
     
     
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/New_data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_1+.pkl'
    temp_list=pd.read_pickle(file_name)
    gain_l_list=[]
    gain_h_list=[]
    
    for i in range(30):
        
        gain_h=0

        
        for t in high_endw:
        
           gain_h+=q.welfare_evaluation(H.G, temp_list[i],t)- q.welfare_evaluation(H.G, {},t)
         
           
        gain_l=0
      
           
        for t  in low_endw:
        
           gain_l+=q.welfare_evaluation(H.G, temp_list[i],t) -q.welfare_evaluation(H.G, {},t)
          
           

        gain_l_list.append(gain_l)
        gain_h_list.append(gain_h)
        
    
    partial_table[l,0]=z
    partial_table[l,1]=l+1
    partial_table[l,2]=(statistics.mean(gain_l_list))
    partial_table[l,3]=(statistics.mean(gain_h_list))


table[0:5,:]=partial_table


z=0.5
partial_table=np.zeros((5,5))

q=Greedy_algorithm_1(50,1,0.8,10,z)

for l in range(5):
     
     
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/New_data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_1+.pkl'
    temp_list=pd.read_pickle(file_name)
    gain_l_list=[]
    gain_h_list=[]
    
    for i in range(30):
        
        gain_h=0

        
        for t in high_endw:
        
           gain_h+=q.welfare_evaluation(H.G, temp_list[i],t)- q.welfare_evaluation(H.G, {},t)
         
           
        gain_l=0
      
           
        for t  in low_endw:
        
           gain_l+=q.welfare_evaluation(H.G, temp_list[i],t) -q.welfare_evaluation(H.G, {},t)
          
           

        gain_l_list.append(gain_l)
        gain_h_list.append(gain_h)
        
    
    partial_table[l,0]=z
    partial_table[l,1]=l+1
    partial_table[l,2]=(statistics.mean(gain_l_list))
    partial_table[l,3]=(statistics.mean(gain_h_list))


table[5:10,:]=partial_table




z=1
partial_table=np.zeros((5,5))

q=Greedy_algorithm_1(50,1,0.8,10,z)

for l in range(5):
     
     
    file_name='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/New_data/reply_alpha_'+str(z)+'_t_.'+ str(l+1) + '_rho_1+.pkl'
    temp_list=pd.read_pickle(file_name)
    gain_l_list=[]
    gain_h_list=[]
    
    for i in range(30):
        
        gain_h=0

        
        for t in high_endw:
        
           gain_h+=q.welfare_evaluation(H.G, temp_list[i],t)- q.welfare_evaluation(H.G, {},t)
         
           
        gain_l=0
      
           
        for t  in low_endw:
        
           gain_l+=q.welfare_evaluation(H.G, temp_list[i],t) -q.welfare_evaluation(H.G, {},t)
          
           

        gain_l_list.append(gain_l)
        gain_h_list.append(gain_h)
        
    
    partial_table[l,0]=z
    partial_table[l,1]=l+1
    partial_table[l,2]=(statistics.mean(gain_l_list))
    partial_table[l,3]=(statistics.mean(gain_h_list))


table[10:15,:]=partial_table


df = pd.DataFrame(table)


file_name_df='C:/Users/CAH259/OneDrive - University of Pittsburgh/Temporal_seeding/R_files/Low_high_1.csv'


df.to_csv(file_name_df,index=False)




    
    






