import sys
from math import acos, asin, cos, pi, sin, sqrt

import inkex
from inkex import TextElement, Line
from inkex.utils import debug

import numpy as np
import math

from svg.path import parse_path

class Hachures(inkex.EffectExtension):

    def add_arguments(self, pars): # Fonction qui récupère les paramètres pour les mettre dans self.options.nom_du_parametre
    
    	# On récupère la liste des dessins sélectionnés
        #self.arg_parser.add_argument("--id", action="append", type=str, dest="ids", default=[], help="id attribute of object to manipulate")
        

        
    	# On résupère les parametres
        pars.add_argument("--periode_distance", type=float, default=10.0, 
            help="Période d'espacement des hachures")
        pars.add_argument("--periode_unite", type=str, default="mm", 
            help="Période d'espacement des hachures")
        pars.add_argument("--angle", type=float, default=45.0,  
            help="Angle des hachures")
        
        
    # =====================================================
    # FONCTION PRINCIPAL
    # =====================================================
    def effect(self): # Fonction qui "modifie" le code SVG
    
        # Éléments utiles ---------------------------
        layer = self.svg.get_current_layer()	# Calque courant
        selection = self.options.ids	# Liste des éléments sélectionnés
        self.style = {'fill' : 'none', 'stroke' : '#000000','stroke-width' : '0.264583'} # Style par defaut
        svg = self.svg # Ref vers l'objet dessin entier
        self.DEBUG = True # Pour moi...

        self.theta = self.options.angle * math.pi/180 # Angle inclinaison hachure (par rapport à l'horizontale)

        # Création du repère penché -------------------------------
        self.ex = np.array([math.cos(self.theta),math.sin(self.theta)]) # Axe parallele aux hachures
        self.ey = np.array([-math.sin(self.theta),math.cos(self.theta)]) # Axe perpendiculare aux hachures
        self.repere = [self.ex,self.ey]
        if(self.DEBUG):
            layer.add(self.traceHachure(0,0,10*self.ex[0], 10*self.ex[1]))
            layer.add(self.traceHachure(0,0,10*self.ey[0], 10*self.ey[1]))


        # Pour chaque figure ,on fait les hachures
        for i in range(len(selection)): # Pour chaque objet sélectionné
            self.debug("Section "+str(i))
            element = svg.getElementById(selection[i])
            chemin = element.to_path_element()

            # On cherche la place qu'il prend (rectangle bounding box)
            Ymin,Ymax = self.getYminYmax(chemin)
            Xmin,Xmax = self.getXminXmax(chemin)

           
            
            objetChemin = parse_path(chemin.get_path())
            self.debug(objetChemin)
            
            for Y in np.arange(Ymin,Ymax+self.options.periode_distance,self.options.periode_distance):
                self.debug("      - Nouvelle hachure")
                P1 = self.ex*Xmin+self.ey*Y
                P2 = self.ex*Xmax+self.ey*Y
                listeIntersections = self.getIntersections(P1,P2,objetChemin)
                self.traceHachureEntreIntersections(layer,listeIntersections)
                #layer.add(self.traceHachure(P1[0],P1[1],P2[0],P2[1]))

        
    # =====================================================
    # FIN PROGRAMME PRINCIPAL
    # =====================================================




















    def debug(self,texte):
        if(self.DEBUG):
            debug(texte)




    def traceHachure(self,x1_,y1_,x2_,y2_):
    	hachure = Line(x1=str(x1_),y1=str(y1_),x2=str(x2_),y2=str(y2_))
    	hachure.style=self.style
    	return hachure
    	
    def log(self,texte):
        elem = TextElement(x="0", y="0")
        elem.text=str(texte)
        elem.style={'font-size': self.svg.unittouu('18pt'),'fill-opacity': '1.0','stroke': 'none','font-weight': 'normal','font-style': 'normal' }
        self.svg.get_current_layer().add(elem)
        
        
    def getYminYmax(self,chemin):
	    # On cherche les abscisses extrême des hachures (dans le repère penché)
	
        bounding_box = chemin.bounding_box() # Bornes de l'espace qu'occupe l'élément
        pointMin = bounding_box.minimum
        pointMax = bounding_box.maximum
        P1 = np.array([pointMin[0],pointMin[1]])
        P2 = np.array([pointMax[0],pointMin[1]])
        P3 = np.array([pointMax[0],pointMax[1]])
        P4 = np.array([pointMin[0],pointMax[1]])
        sommets = [P1,P2,P3,P4]
           
        yMin = 1000000
        yMax = -1000000
        for i in range(4):
            proj = np.dot(sommets[i],self.ey)
            if(proj<yMin):
             yMin=proj
            if(proj>yMax):
                yMax=proj
                
        yMin = (yMin//self.options.periode_distance)*self.options.periode_distance # Décallage (pour être sûr de démarrer AVANT et finir APRES la figure)
        yMax = (yMax//self.options.periode_distance+1)*self.options.periode_distance
        return (yMin,yMax)
        
    def getXminXmax(self,chemin):
	    # On cherche les abscisses extrême des hachures (dans le repère penché)
	
        bounding_box = chemin.bounding_box() # Bornes de l'espace qu'occupe l'élément
        pointMin = bounding_box.minimum
        pointMax = bounding_box.maximum
        P1 = np.array([pointMin[0],pointMin[1]])
        P2 = np.array([pointMax[0],pointMin[1]])
        P3 = np.array([pointMax[0],pointMax[1]])
        P4 = np.array([pointMin[0],pointMax[1]])
        sommets = [P1,P2,P3,P4]
           
        xMin = 1000000
        xMax = -1000000
        for i in range(4):
            proj = np.dot(sommets[i],self.ex)
            if(proj<xMin):
             xMin=proj
            if(proj>xMax):
                xMax=proj
        xMin -= 1 # Pour dépasser un peu (et être sûr d'avoir les intersections avec les bords)
        xMax += 1 
        return (xMin,xMax)
          
          
          
#    def explosePath(self,path):
#        decoupe = path.split(" ")
#        liste_de_chemins = [] # Liste qui contiendra les points de chaque chemin (s'il y a plusieurs chemins...)
#        chemin = [] # Un chemin
#        liste_de_chemins.append(chemin)
#
#        for i in range(len(decoupe)):
#            if decoupe[i] in ["m" , "M" ] : # Si on a bougé sans tracer
#                chemin=[[decoupe[i],float(decoupe[i+1]),float(decoupe[i+2])]]
#                liste_de_chemins.append(chemin) # On range le précédent chemin
#                i+=2
#            elif decoupe[i] in ["l" ,"L"]:
#                chemin.append([decoupe[i],float(decoupe[i+1]),float(decoupe[i+2])])
#                i+=1
#            elif decoupe[i] in ["h","H"]:
#               chemin.append([decoupe[i],chemin[-2],float(decoupe[i+1])])
#               i+=1
#            elif decoupe[i] in ["v","V"]:
#               chemin.append([decoupe[i],float(decoupe[i+1]),chemin[-2]])
#               i+=1
#            elif decoupe[i] in ["z","Z"]:
#                chemin.append(chemin[0])
#                i+=1
#            elif decoupe[i] in ["c","C"]:
#                chemin.append([decoupe[i],float(decoupe[i+1]), float(decoupe[i+2]), float(decoupe[i+5]), float(decoupe[i+6]) ])
#                chemin.append([decoupe[i],float(decoupe[i+3]), float(decoupe[i+4])])
#                i+=6
#            
#            if liste_de_chemins[0]==[]:
#                liste_de_chemins.pop(0)
#            
#        return liste_de_chemins
 
 
 
    def getIntersections(self,P1,P2,objetPath):
        intersections = []
        for troncon in objetPath: # Pour chaque troncon, selon ce que c'est (Ligne, Bezier, etc.)
            debug("   - nouveau troncon !!!!!!!!!!"+str(troncon))
            if(type(troncon).__name__ in ["Line","Close","Arc", "QuadraticBezier"]):
                debug("      - Ligne")
                PA = np.array([troncon.start.real, troncon.start.imag])
                PB = np.array([troncon.end.real, troncon.end.imag])
                if(self.seCroisent(P1,P2,PA,PB)):
                    intersections.append(self.getIntersection(P1,P2,PA,PB))
                    debug("         --> trouvé ")
            elif(type(troncon).__name__ == "CubicBezier"):
                debug("      - Bezier")
                # equation de la droite : ax+by=d
                normale = np.array([[0,-1],[1,0]])@ np.transpose(P2-P1)
                normale = normale/np.linalg.norm(normale)
                a,b = normale
                d = np.dot(P1,normale)
                #Résolution intersection bezier
                #https://math.stackexchange.com/questions/2347733/intersections-between-a-cubic-b%C3%A9zier-curve-and-a-line
                P_0 = np.array([troncon.start.real,troncon.start.imag])
                P_1 = np.array([troncon.control1.real,troncon.control1.imag])
                P_2 = np.array([troncon.control2.real,troncon.control2.imag])
                P_3 = np.array([troncon.end.real,troncon.end.imag])
                f = lambda t:(1-t)**3*(a*P_0[0]+b*P_0[1]) + 3*t*(1-t)**2*(a*P_1[0]+b*P_1[1]) + 3*t**2*(1-t)*(a*P_2[0]+b*P_2[1]) + t**3*(a*P_3[0]+b*P_3[1])-d
                #dichotomie
                tmin=-0.2
                tmax=1.2
                tmid = 0.5*(tmin+tmax)
                while abs(tmin-tmax)>0.001:
                    if f(tmid)*f(tmin)>0:
                        tmin=tmid
                    else:
                        tmax=tmid
                    tmid = 0.5*(tmin+tmax)
                    #debug(tmid)
                if(tmid>=0 and tmid<=1):   # Si on est dans la courbe de Bezier (dans [0,1])
                    intersections.append((1-tmid)**3*P_0 + 3*tmid*(1-tmid)**2*P_1 + 3*tmid**2*(1-tmid)*P_2 + tmid**3*P_3)
                    debug("        -> trouvé ")
                
        debug(intersections)
                    
        # On tri les intersection
        for i in range(len(intersections)-1):
            for j in range(len(intersections)-1):
                if(self.distanceALorigineParallelementHachure(intersections[j]) > self.distanceALorigineParallelementHachure(intersections[j+1])):
                    intersections[j],intersections[j+1] = intersections[j+1],intersections[j]
        return intersections
        
    def distanceALorigineParallelementHachure(self,P):
        return np.dot(self.ex,P)

                    
    
    def seCroisent(self,P1,P2,PA,PB): # Ce qu'il y a près le "and" n'est peut être pas obligatoire (si les hachures prolongent à l'infini
        #P1 et P2 = extrémités max de la hachure.
        # PA et PB = extremités du segment
        if(np.cross((P2-P1),(PA-PB))==0):
            return False
        return (np.cross((P1-P2),(PA-P1))*np.cross((P1-P2),(PB-P1))<0) #and (np.cross((PB-PA),(P2-PA))*np.cross((PB-PA),(P1-PA))<0)
      
    def getIntersection(self,P1,P2,PA,PB):
        # Résultat du système : P1C ^ P1P2 = 0 et PAC ^ PAPB = 0
        x1,y1,x2,y2 = P1[0],P1[1] , P2[0],P2[1]
        xA,yA,xB,yB = PA[0],PA[1] , PB[0],PB[1]
        
        yC = (-x1*(y2-y1)*(yB-yA)+y1*(x2-x1)*(yB-yA)+xA*(yB-yA)*(y2-y1)-yA*(xB-xA)*(y2-y1))/((x2-x1)*(yB-yA)-(xB-xA)*(y2-y1))
        xC = (yC*(xB-xA)+xA*(yB-yA)-yA*(xB-xA))/(yB-yA)
        return np.array([xC,yC])
        
        
    # Fonction qui trace les hachures, une intersection sur deux
    def traceHachureEntreIntersections(self,layer,inter):
        if(len(inter)>1):
            for i in range(0,len(inter),2):
                if(i < len(inter)-1):
                    P1 = inter[i]
                    P2 = inter[i+1]
                    layer.add(self.traceHachure(P1[0],P1[1],P2[0],P2[1]))
                    
                    
                    
if __name__ == '__main__':
    Hachures().run()
