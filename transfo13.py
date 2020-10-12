import numpy as np
import math as ma
import random
import copy
import time
import matplotlib.pyplot as plt


""" vol = [tour de passage, cout,numéro de vol]
plan_de_vol=[V00,V01,...]
"""


def generateur_plan_de_vol(nombre_de_vols_total,cout_max):
    """génére aléatoirement un plan de vol"""
    resu=[]
    for x in range(nombre_de_vols_total):
        resu.append([random.randint(1,nombre_de_vols_total),random.randint(1,cout_max)])
    resu=tri_fusion(resu)
    for i in range(len(resu)):
        resu[i].append(i)
    return resu
    
def fusion(li,la):
    if li==[]:
        return la
    if la==[]:
        return li
    if li[0][0]<la[0][0]:
        return [li[0]]+fusion(li[1:],la)
    elif li[0][0]==la[0][0]:
        return[li[0],la[0]]+fusion(li[1:],la[1:])
    else:
        return [la[0]]+fusion(la[1:],li)

def tri_fusion_rec_aux(ma_liste,i1,i2):
    if i2<=i1+1:
        return ma_liste[i1:i2]
    else:
        im=(i1+i2)//2
        return fusion(tri_fusion_rec_aux(ma_liste,i1,im),tri_fusion_rec_aux(ma_liste,im,i2))
        
def tri_fusion(ma_liste):
    n=len(ma_liste)
    return tri_fusion_rec_aux(ma_liste,0,n)

def matrice_conflit(plan_de_vol):
    """ cette fonction permet de calculer la matrice conflit de l'ensemble des décollages et aterrisages"""
    Mat = np.zeros([len(plan_de_vol),len(plan_de_vol)])
    for i in range(len(plan_de_vol)):
        for j in range(len(plan_de_vol)):
            if plan_de_vol[i][0] == plan_de_vol[j][0]: 
                Mat[i][j] += 1
    return Mat


def enlever_element(li,x):
    """enleve un element de la liste li""" 
    la=li[:]
    la.remove(x)
    return la


def ajout_graphe(p,li,z,mat,deb1,pioche,compteur,reference,plan_de_vol):
    """ C'est un algorithme récursif qui permet d'ajouter l'ensemble des combinaisons de vols possibles (sans conflit) dans le graphe, c'est l'algorithme central du sujet, elle permet de construire l'ensemble des solutions de vols possibles sous forme d'arbre"""
    if li==[]: #si la liste est vide c'est le dernier élément
        deb1.append(p)
    elif compteur==0:
        for x in li:
            prise=pioche.pop()
            reference[x].append(prise)
            mat[p][prise]=1
            ajout_graphe(prise,enlever_element(li,x),z+1,mat,deb1,pioche,compteur+1,reference,plan_de_vol)
    else:
        for x in li:
            prise=pioche.pop()
            reference[x].append(prise)
            mat[p][prise]=z*plan_de_vol[x][1]
            ajout_graphe(prise,enlever_element(li,x),z+1,mat,deb1,pioche,compteur+1,reference,plan_de_vol)
            


def transfo_matrice(mconf,long_mat,plan_de_vol):
    """Cet algorithme construit au moyen des algorithmes précédents la matrice d'adjacence, du graphe de l'ensemble des vols possibles"""

    mat=np.zeros([long_mat,long_mat])
    pioche=[n for n in range(1000000,0,-1)]
    reference=[[] for n in range (len(plan_de_vol))]
    compteur=0
    deb=[0] # deb est la liste qui contient les dernier éléments du graphe à t
    deb1=[] # deb1 est un outil qui permet de modifier deb pour le prochain tour
    graph=[] # graph permet de connaitre les éléments déjà présent sur le graphe
    z=0    # z permet d'éviter qu il y ait des noeuds qui ont le meme numéro
    for i in range(len(plan_de_vol)):
        if i not in graph: # on évite ainsi les cas ou i est déjà dans le graphe
            list=[]
            for j in range(len(plan_de_vol)):
                if mconf[i][j]!=0 and (j not in graph): # y a pas conflit si le vol est deja dans le graphe
                    list.append(j)
            if list!=[i]:
                for p in deb:
                    ajout_graphe(p,list,z,mat,deb1,pioche,compteur,reference,plan_de_vol) # cette fonction place les elements sur le graphe
                z+=len(list)
                compteur+=1
                (deb,deb1)=(deb1,[]) # permet le passage du tour t au tour t+1 deb1 et une liste de passage    
            else: #quand il n y a pas de conflit il faut quand meme ajouter le vol sur le graphe pour pouvoir trouver ...                
                ind=pioche.pop()
                reference[i].append(ind)
                for p in deb: # le plus court chemin, car il faut qu il y ait un chemin, on ajoute donc 1, cela ne change rien"""
                    mat[p][ind]=1
                compteur+=1
                deb=[ind] # à ce moment le dernier element du graphe est i
            for x in range(len(plan_de_vol)):
                if mconf[i][x]==0:
                    plan_de_vol[x][0]=plan_de_vol[x][0]+z # on décale tout le monde, cela ne change rien 
            mconf=matrice_conflit(plan_de_vol) 
            graph=list+graph
    for x in deb: # on lie les derniers vols a l'état final
        mat[x][long_mat-1]=1
    return (mat,reference)
                        
        
def appartenance(convert,list,plan_de_vol):
    """ utilisé pour passer du chemin suivi selon les numéros du graphe au chemin suivi selon les numéros de vols"""
    resu=[]
    for x in list:
        for i in range(len(convert)):
            if x in convert[i]:
                resu.append(i)
    print("COUT TOTAL =",cout_final(resu,plan_de_vol))
    print('VOLS SUIVI',';','->'.join(map(str,resu)))            


def cout_final(plan_final,plan_de_vol):
    """permet de calculer le cout final"""
    resu=0
    compteur=0
    for x in plan_final:
        for y in plan_de_vol:
            if x==y[-1]:
                resu+= abs(y[0]-compteur-1)*y[1]
                compteur+=1
    return resu


def matriceDijkstra (mg,convert,s,long_mat,plan_de_vol):
    """ Cet algorithme prend en argument une matrice d'adjacence, et un point de départ et calcule le plus court chemin pour aller d'un point à l'autre avec la méthode de Dijkstra.Elle nous permet de trouver la solution optimale de plan de vols en prenant en argument la matrice d'adjacence calculée grâce à l'algorithme précédent"""
    infini= ma.inf
    nb_sommets= len(mg)
    s_connu = {s:[0,[s]]} # on construit les chemins déja présents
    s_inconnu = {k:[infini,''] for k in range (nb_sommets) if k !=s}#et les chemins à explorer
    for suivant in range(nb_sommets):
        if mg [s][suivant]:
            s_inconnu[suivant]=[mg[s][suivant],s]
    print('Le plus court chemin de l état initial à l état final est')
    while s_inconnu and any(s_inconnu[k][0]<infini for k in s_inconnu):
        u=min(s_inconnu,key=s_inconnu.get)
        longueur_u,precedent_u=s_inconnu[u]
        for v in range (nb_sommets):
            if mg[u][v] and v in s_inconnu:
                d= longueur_u+ mg[u][v]
                if d< s_inconnu[v][0]:
                    s_inconnu[v]=[d,u]
        s_connu[u]=[longueur_u,s_connu[precedent_u][1]+[u]]
        del s_inconnu[u]
        if s_connu[u][1][-1]==(long_mat-1):
            print(s_connu[u][1])
            print("sur le graphique:")
            print('longueur',longueur_u,';','->'.join(map(str,s_connu[u][1])))
            print("")
            print(appartenance(convert,s_connu[u][1],plan_de_vol))
    
    


def fonction_finale (plan_de_vol,long_mat):
    """la fonction finale"""
    print(plan_de_vol)
    plan_de_vol_bis=copy.deepcopy(plan_de_vol)
    final=transfo_matrice(matrice_conflit(plan_de_vol),long_mat,plan_de_vol)
    return matriceDijkstra(final[0],final[1],0,long_mat,plan_de_vol_bis)

###########################################################################################################

""" Fonctions d'études statistiques"""


def test(nombre_de_vols,long_mat,nombre_test):
    """effectue des l'algorithme n fois"""
    tps=[]
    for k in range(nombre_test):
        plan=generateur_plan_de_vol(nombre_de_vols,random.randint(50,100))
        print(plan)
        alpha_pdv=copy.deepcopy(plan)
        start = time.time() 
        fonction_finale(alpha_pdv,long_mat)
        end = time.time()
        tps.append(end - start)
    return np.mean(tps)
        
                        
        
def graphique_test_temporel_nombre_de_vols(nombre_maximale_de_vols,longueur_matrice,nombre_test_par_plan):
    """construit le graphe des études"""
    X=[x for x in range(1,nombre_maximale_de_vols+1)]
    Y=[]
    fi = open('/Users/as_akin/Desktop/TIPE_stat/stat_test_temporel_nombre_de_vols.txt','w')
    for i in range(1,nombre_maximale_de_vols+1):
        longueur_matrice+=5000
        val=test(i,longueur_matrice,nombre_test_par_plan)
        Y.append(test(i,longueur_matrice,nombre_test_par_plan))
        fi.write(str(val)+'\n')
    fi.close()
    plt.xlabel('nombre de vols')
    plt.ylabel('durée éxecution')
    plt.title('Etude de la complexité temporelle en fonction du nombre de vols')
    plt.plot(X,Y,'r')
    plt.show()
        

def graphique_test_temporel_long_matrice(nombre_de_vols,longueur_maximale_matrice,longueur_mnimale_matrice,nombre_test_par_plan):
    """étudie l'influence de long_matrice"""
    if longueur_minimale_matrice<nombre_de_vols*100:
        print("Cette valeur de longueur peut-être trop petite par rapoort au nombre de vols choisis")
    else:
        X=[x for x in range(longueur_minimale_matrice,longueur_maximale_matrice+1)]
        Y=[]
        for i in range(longueur_minimale_matrice,longueur_maximale_matrice+1):
            Y.append(test(i,longueur_maximale_matrice,nombre_test_par_plan))
        fi = open('/Users/as_akin/Desktop/TIPE_stat/stat_test_temporel_long_matrice.txt','w')
        for k in range(len(Y)):
            fi.write(str(Y[k])+'\n')
        fi.close()
        plt.xlabel('nombre de vols')
        plt.ylabel('durée éxecution')
        plt.title('Etude de la complexité temporelle en fonction de la longueur de la matrice')
        plt.plot(X,Y,'r')
        plt.show()
        

def test_dijkstra(nombre_de_vols,long_mat,nombre_test):
    tps=[]
    tps_tot=[]
    for k in range(nombre_test):
        plan=generateur_plan_de_vol(nombre_de_vols,random.randint(50,1000))
        alpha_pdv=copy.deepcopy(plan)
        start_1=time.time()
        final=transfo_matrice(matrice_conflit(alpha_pdv),long_mat,alpha_pdv)
        end_1 = time.time()
        start_2 =time.time()
        matriceDijkstra(final[0],final[1],0,long_mat,alpha_pdv)
        end_2 = time.time()
        tps.append(end_2 - start_2)
        tps_tot.append((end_1-start_1)+(end_2 - start_2))
    return (np.mean(tps),np.mean(tps_tot))
    


def graphique_test_temporel_dijkstra(nombre_maximale_de_vols,longueur_matrice,nombre_test_par_plan):
    X=[x for x in range(1,nombre_maximale_de_vols+1)]
    Y1=[]
    Y2=[]
    for i in range(1,nombre_maximale_de_vols+1):
        test=test_dijkstra(i,longueur_matrice,nombre_test_par_plan)
        Y1.append(test[0])
        Y2.append(test[1])
    fi = open('/Users/as_akin/Desktop/TIPE_stat/stat_test_temporel_dijkstra_dij.txt','w')
    for k in range(len(Y1)):
        fi.write(str(Y1[k])+'\n')
    fi.close()
    fi = open('/Users/as_akin/Desktop/TIPE_stat/stat_test_temporel_dijkstra_tot.txt','w')
    for k in range(len(Y2)):
        fi.write(str(Y2[k])+'\n')
    fi.close()
    plt.xlabel('nombre de vols')
    plt.ylabel('durée éxecution')
    plt.title('Etude de la complexité temporelle en fonction du nombre de vols')
    plt.plot(X,Y1,'r',X,Y2,'b')
    plt.legend(['dijkstra','algorithme entier'])
    plt.show()
    

def test_transfo_matrice(nombre_de_vols,long_mat,nombre_test):
    tps=[]
    tps_tot=[]
    for k in range(nombre_test):
        plan=generateur_plan_de_vol(nombre_de_vols,random.randint(50,1000))
        alpha_pdv=copy.deepcopy(plan)
        start_1=time.time()
        final=transfo_matrice(matrice_conflit(alpha_pdv),long_mat,alpha_pdv)
        end_1 = time.time()
        start_2 =time.time()
        matriceDijkstra(final[0],final[1],0,long_mat,alpha_pdv)
        end_2 = time.time()
        tps.append(end_1 - start_1)
        tps_tot.append((end_1-start_1)+(end_2 - start_2))
    return (np.mean(tps),np.mean(tps_tot))


fi = open('/Users/as_akin/Desktop/TIPE_stat/stat_test_temporel_transfo_matrice_trans.txt','w')
fit = open('/Users/as_akin/Desktop/TIPE_stat/stat_test_temporel_transfo_matrice_tot.txt','w')


def graphique_test_temporel_transfo_matrice(nombre_maximale_de_vols,longueur_matrice,nombre_test_par_plan):
    X=[x for x in range(1,nombre_maximale_de_vols+1)]
    Y1=[]
    Y2=[]
    for i in range(1,nombre_maximale_de_vols+1):
        test=test_dijkstra(i,longueur_matrice,nombre_test_par_plan)
        Y1.append(test[0])
        Y2.append(test[1])
        fi.write(str(test[0])+'\n')
        fit.write(str(test[1])+'\n')
    plt.xlabel('nombre de vols')
    plt.ylabel('durée éxecution')
    plt.title('Etude de la complexité temporelle en fonction du nombre de vols')
    plt.plot(X,Y1,'r',X,Y2,'b')
    plt.legend(['transfo_matrice','algorithme entier'])
    plt.show()
        


