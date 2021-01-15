# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 17:02:36 2021

@author: phant
"""
#Importation des libraries
import urllib.request
import xmltodict  
import pandas 
import numpy
import matplotlib.pyplot as plt
import itertools
import networkx as nx
from pyvis.network import Network
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import tkinter as tk
from community import community_louvain

################################## Interface Tkinter ##################################

#Création de l'interface Tkinter
root= tk.Tk()
canvas1 = tk.Canvas(root, width = 1000, height = 500)
canvas1.pack()

#Création d'un label pour l'interface
label1 = tk.Label(root, text="Construction d'un graphe de cooccurrence")
label1.config(font=('helvetica', 16))
canvas1.create_window(500, 25, window=label1)

#Création d'un label pour l'interface
label2 = tk.Label(root, text="à partir d'un corpus d'articles")
label2.config(font=('helvetica', 16))
canvas1.create_window(500, 50, window=label2)

#Création d'un label pour l'interface
label3 = tk.Label(root, text= "Veuillez rechercher un thème de corpus et choisir un nombre d'articles:")
canvas1.create_window(500, 100, window=label3)

#Création d'un label pour l'interface
label4 = tk.Label(root, text= "Thème: ")
canvas1.create_window(350, 140, window=label4)

#Input box pour le thème du corpus
entry1 = tk.Entry (root) 
canvas1.create_window(500, 140, window=entry1)

#Création d'un label pour l'interface
label5 = tk.Label(root, text= "Nombre d'articles: ")
canvas1.create_window(350, 170, window=label5)

#Input box pour le nombre d'articles dans le corpus
entry2 = tk.Entry (root) 
canvas1.create_window(500, 170, window=entry2)

#######################################################################################

################################## Fonction principale du programme: Création du graphe de cooccurrence ##################################

#Fonction graphe_co_occurrence. Cette dernière permet, à partir d'un corpus d'articles, de produire un graphique des cooccurrences des mots
def graphe_cooccurrence():

################################## Importation ##################################

    #Importation des données vi l'API de arxiv
    corpus = []
    #input=la valeur choisie par l'utilisateur
    input = entry1.get()
    input2 = entry2.get()
    #url de arxiv dans laquelle on insère le choix de sujet de l'utilisateur
    url = "http://export.arxiv.org/api/query?search_query=all:"+input+"&start=0&max_results="+input2
    data =  urllib.request.urlopen(url).read().decode()
    # decode() transforme le 'bytes stream" en chaîne de caractères
    docs = xmltodict.parse(data)['feed']['entry']
    for i in docs: # docs est une liste de "dictionnaires ordonnés"
        print(i['title'])
        txt = i['summary']
        txt = txt.replace('\n', ' ')
        corpus.append(txt)
    
    #On vérifie la longueur du corpus
    len(corpus)

##################################################################################

############################## Nettoyage du corpus ###############################

    #On converti corpus en dataframe afin de pouvoir le transformer en liste numpy
    df = pandas.DataFrame(corpus)
    articles=df[0].str.lower().to_numpy()
    
    #Tokenizer permet d'enlever les caractères inutiles comme les virgules par exemple
    tokenizer = RegexpTokenizer(r'\w+')
    #stop_words contient les mots anglais qui ne nous intéresse pas
    stop_words = set(stopwords.words('english'))
    
    #Pour chaque articles du corpus, on va le nettoyer des caractères inutiles
    article_mots = []
    for article in articles:
        mots=[]
        #Pour chaque mots, si sa longueur est >3 et qu'il n'est pas dans la liste des mots inutiles, on l'ajoute à article_mots
        for token in tokenizer.tokenize(article):
            if len(token) > 3 and token not in stop_words:
                mots.append(token)
        article_mots.append(mots)
    
###################### Création de la liste de vocabulaire #######################
    
    #On compte l'occurence de chaque mots de chaque articles
    nb_mots = {}
    for mots in article_mots:
        for mot in mots:
            if mot not in nb_mots:
                nb_mots[mot] = 1
            else:
                nb_mots[mot] += 1
    
    #On insère dans un dataframe les mots et leurs occurrences
    nb_mots_df = pandas.DataFrame({'mots': [k for k in nb_mots.keys()], 'occurences': [v for v in nb_mots.values()]})
    
    #On plot un graphique qui montre les mots qui ont les plus grandes occurrences
    #Nous avons décidé de garder seulement les mots qui ont une longueur supérieures au tier de la longueur des articles
    #Ainsi, les graphiques ne sont pas surchargés
    tmp = nb_mots_df[nb_mots_df['occurences'] > int(len(article_mots)/3)]
    tmp.sort_values(by='occurences', ascending=False)
    tmp.plot(kind='bar', x='mots', y='occurences', figsize=(15,7), legend=False, title='Mots par occurence')
    plt.show()
    
    #On crée la liste de vocabulaire en fonction du nombre d'occurence des mots
    vocab_list = {}
    
    #Pareil que pour le graphique, les mots ajoutés au vocabulaire sont sélectionnés
    mots_filtrés = nb_mots_df[nb_mots_df['occurences'] > int(len(article_mots)/3)]['mots'].to_numpy()
    for mot in mots_filtrés:
        if mot not in vocab_list:
            vocab_list[mot] = len(vocab_list)
    
    #Par soucis de modélisation, nous inversons les mots et leurs occurrences
    re_vocab = {}
    for mot, i in vocab_list.items():
        re_vocab[i] = mot

##################################################################################

#################### Création de la matrice de cooccurrence #####################
    
    #On créé toute les paires de 2 mots possibles dans chaque articles
    co_occurrences = [list(itertools.combinations(mots, 2)) for mots in article_mots]
    #On l'a crée vide de longueur i et j = longueur de la liste de voca
    matrice_cooccurrence = numpy.zeros((len(vocab_list), len(vocab_list)))
    
    #Création de la matrice
    for co_occurrence in co_occurrences:
        for co_occ in co_occurrence:
            if  co_occ[0] in mots_filtrés and  co_occ[1] in mots_filtrés:
                matrice_cooccurrence[vocab_list[co_occ[0]], vocab_list[co_occ[1]]] += 1
                matrice_cooccurrence[vocab_list[co_occ[1]], vocab_list[co_occ[0]]] += 1
            
    #On divise la diagonale par deux car on les incrémentes deux fois trop
    for i in range(len(vocab_list)):
        matrice_cooccurrence[i, i] /= 2
    
    #Print de la matrice de cooccurence avec les colonnes
    print(pandas.DataFrame(data=matrice_cooccurrence, columns=vocab_list,index=vocab_list))
    
    #Par soucis d'affichage, on divise la matrice par 100. En effet, Sinon les arcs du graphes sont énormes
     #matrice_cooccurrence=matrice_cooccurrence/100
       
##################################################################################

######################## Création de noeuds et des arcs ##########################

    #On créé la liste vide noeuds
    noeuds = []
    #On parcours la matrice. Si une valeur est différente de 0 alors on ajoutes les mots, les occurrences des mots ainsi que le poids de l'arc dans noeud
    for i in range(len(vocab_list)):
        for j in range(i+1, len(vocab_list)):
            co = matrice_cooccurrence[i, j]
            if co > 0:
                noeuds.append([re_vocab[i], re_vocab[j], nb_mots[re_vocab[i]], nb_mots[re_vocab[j]], co])
            
##################################################################################

############################ Création du graphique ###############################

    #On initialise le graphique G qui servira de base pour la partition
    G = nx.Graph()
    G.nodes(data=True)
    for paire in noeuds:
        noeuds_x, noeuds_y, noeuds_x_cnt, noeuds_y_cnt, co = paire[0], paire[1], paire[2], paire[3], paire[4]
        if not G.has_node(noeuds_x):
            G.add_node(noeuds_x, 
                       count=noeuds_x_cnt, 
                       size=noeuds_x_cnt, 
                       color="red",
                       title= "Mot: " + noeuds_x + " ;\n" + str(noeuds_x_cnt) + " Occurrences")
        if not G.has_node(noeuds_y):
            G.add_node(noeuds_y, 
                       count=noeuds_y_cnt,
                       size=noeuds_y_cnt,
                       color="red",
                       title= "Mot: " + noeuds_y + " ;\n" + str(noeuds_y_cnt) + " Occurrences")
        if not G.has_edge(noeuds_x, noeuds_y):
            G.add_edge(noeuds_x,
                       noeuds_y,
                       weight=co,
                       color="grey",
                       title="Cooccurrence: " + str(co))

    #Mise en cluster pour créer les communautés
    partition = community_louvain.best_partition(G)
    
    #On initialise le graphique K
    #Création des noeuds et des arcs dans le graphique. On personnalise aussi la couleur en fonction de la partition et on ajoute des titres aux noeuds
    K = nx.Graph()
    K.nodes(data=True)
    for paire in noeuds:
        noeuds_x, noeuds_y, noeuds_x_cnt, noeuds_y_cnt, co = paire[0], paire[1], paire[2], paire[3], paire[4]
        if not K.has_node(noeuds_x):
            part=partition.get(noeuds_x, 0)
            K.add_node(noeuds_x, 
                       count=noeuds_x_cnt, 
                       size=noeuds_x_cnt, 
                       group=part,
                       title= "Mot: " + noeuds_x + " ;\n" + str(noeuds_x_cnt) + " Occurrences")
        if not K.has_node(noeuds_y):
            part2=partition.get(noeuds_y, 0)
            K.add_node(noeuds_y, 
                       count=noeuds_y_cnt,
                       size=noeuds_y_cnt,
                       group=part2,
                       title= "Mot: " + noeuds_y + " ;\n" + str(noeuds_y_cnt) + " Occurrences")
        if not K.has_edge(noeuds_x, noeuds_y):
            K.add_edge(noeuds_x,
                       noeuds_y,
                       weight=co,
                       color="grey",
                       title="Cooccurrence: " + str(co))
            
    
    #Cette partie sert à créer la page HTML qui acceuillera le graphique
    nt = Network("700px", "1450px",heading="Graphique de cooccurrence")
    nt.barnes_hut()
    nt.from_nx(K)
    
    #On rajoute de nouveaux labels à l'interface
    label6 = tk.Label(root, text= "Le graphe s'est ouvert dans votre navigateur!",font=('helvetica', 10))
    canvas1.create_window(500, 230, window=label6)
    label7 = tk.Label(root, text= "Vous pouvez refaire une recherche si vous le désirez!",font=('helvetica', 10))
    canvas1.create_window(500, 250, window=label7)

##################################################################################

    
##################### Identifications des noyaux centraux ########################

    #degrées de centralité
    #Pour récupérer le nom et le degré correspondant, on a mis les résultats de K.degree() dans un numpy.array
    valeur={'name':numpy.array(K.degree())[:,0],'degree':numpy.array(K.degree())[:,1]}
    degree=pandas.DataFrame.from_dict(valeur).sort_values(by='degree',ascending=False).iloc[:5,:]

    #closeness centralité
    valeurs={'closeness':nx.closeness_centrality(K)}
    closeness=pandas.DataFrame.from_dict(valeurs).sort_values(by='closeness',ascending=False).iloc[:5,:]
    

    #betwenness centralité
    valeurss={'betweenness':nx.betweenness_centrality(K)}
    betweenness=pandas.DataFrame.from_dict(valeurss).sort_values(by='betweenness',ascending=False).iloc[:5,:]
    
    
    #On rajoute de nouveaux labels à l'interface mais avec les tableaux de centralité
    label11 = tk.Label(root, text="Identification des 5 noeuds centraux selon différents critères")
    label11.config(font=('helvetica', 14))
    canvas1.create_window(500, 280, window=label11)
    
    label8 = tk.Label(root, text= "Degrées de centralité",font=('helvetica', 10))
    canvas1.create_window(200, 310, window=label8)
    text1 = tk.Text(height=8,width=25)
    text1.insert(tk.END, degree)
    canvas1.create_window(200, 390, window=text1)
    
    label9 = tk.Label(root, text= "Centralité Closeness",font=('helvetica', 10))
    canvas1.create_window(500, 310, window=label9)
    text1 = tk.Text(height=8,width=25)
    text1.insert(tk.END, closeness)
    canvas1.create_window(500, 390, window=text1)
    
    label10 = tk.Label(root, text= "Centralité Betweenness",font=('helvetica', 10))
    canvas1.create_window(800, 310, window=label10)
    text1 = tk.Text(height=8,width=25)
    text1.insert(tk.END, betweenness)
    canvas1.create_window(800, 390, window=text1)

##################################################################################


############################# Return du graphique ################################
 
    #On return le lien HTML du graphique
    return (nt.show("Graphique des cooccurrences.html"))

##################################################################################

###########################################################################################################################################

################################## Fin de l'interface Tkinter ##################################

button1 = tk.Button(text='Lancer la recherche', command=graphe_cooccurrence)
canvas1.create_window(500, 200, window= button1)
root.mainloop()

################################################################################################