import sys
from math import acos, asin, cos, pi, sin, sqrt

import inkex
from inkex import TextElement, Line
import inkex.paths as ip
from inkex.utils import debug
from inkex import Path, PathElement

import numpy as np
import math


from svg.path import parse_path

class Hachures(inkex.EffectExtension):

    def add_arguments(self, pars): # Fonction qui récupère les paramètres pour les mettre dans self.options.nom_du_parametre
    
    	# On récupère la liste des dessins sélectionnés
        #self.arg_parser.add_argument("--id", action="append", type=str, dest="ids", default=[], help="id attribute of object to manipulate")
        
    	# On résupère les parametres
    	# Parametres communs
        pars.add_argument("--periode", type=float, default=5.0, help="Période d'espacement des hachures principales")
        pars.add_argument("--offset", type=float, default=0.0, help="Décallage des hachures")
        pars.add_argument("--unite", type=str, default="mm", help="Unité des longueurs")
        pars.add_argument("--angle", type=float, default=45.0, help="Orientation des hachures")
        pars.add_argument("--epaisseur", type=float, default=0.25, help="Épaisseur du trait")
        pars.add_argument("--couleur", type=int, default=0, help="Couleur des lignes")
        pars.add_argument("--parametres_generaux", type=str, default="parametre_dimensions", help="Onglet des paramètres généraux")
        pars.add_argument("--groupe_figure", type=str, default="combo", help="Type de groupement des hachures")
        
        pars.add_argument("--type_materiau", type=str, default="acier", help="Type de hachures")
        pars.add_argument("--longueur_tiret_cuivre", type=float, default=50, help="Pourcentage longueur tiret")
        pars.add_argument("--longueur_espace_cuivre", type=float, default=50, help="Pourcentage longueur espace entre tirets")
        pars.add_argument("--espace_aluminium", type=float, default=50, help="Pourcentage d'esapce espace entre hachures proches")
        pars.add_argument("--angle_plastique", type=float, default=45, help="Angle les hachures du plastique")
        
        
    # =====================================================
    # FONCTION PRINCIPAL
    # =====================================================
    def effect(self): # Fonction qui "modifie" le code SVG
    
        self.DEBUG = False # Pour moi...
        
        #Conversion des couleurs dans un format correct
        self.options.couleur = self.convertIntColor2Hex(self.options.couleur)
        # Conversion des unités
        self.options.periode = self.svg.unittouu(str(self.options.periode)+self.options.unite);
        self.options.offset = self.svg.unittouu(str(self.options.offset)+self.options.unite);
        self.options.epaisseur = self.svg.unittouu(str(self.options.epaisseur)+self.options.unite);
    
        # Éléments utiles ---------------------------
        svg = self.svg # Ref vers l'objet dessin entier
        layer = self.svg.get_current_layer()	# Calque courant
        
        # Conversion en chemin (par exemple rectangle --> chemin, ellipse --> chemin, etc.)
        listeObjetsSelectionnes = []
        for i in range(len(self.options.ids)):
            idElementBrute = self.options.ids[i]
            elementBrute = svg.getElementById(idElementBrute).copy()
            # Petit pb : les rectangles et les ellipses ne sont pas dans la bonne unité : on va appliquer une échelle pour les ramener dans la bonne unité
            #if(elementBrute.tag == inkex.addNS('rect', 'svg')):
            #    self.debug("C'est un rectangle !!!")
            #    self.updateCoordonneesRectangleDansLaBonneUnite(elementBrute)
            elementConvertiEnChemin = elementBrute.to_path_element()
            elementConvertiEnChemin.apply_transform() # On "met à plat" les transformations, directement sur les coordonnées des noeuds
            listeObjetsSelectionnes.append(elementConvertiEnChemin)
        #selection = [svg.getElementById(self.options.ids[i]).to_path_element() for i in range(len(self.options.ids))]	# Liste des éléments sélectionnés 
        
        
        
        
        #self.debug(svg.getElementById(self.options.ids[0]).get_path())
        #self.debug(listeObjetsSelectionnes[0].get_path())
        
        # Fusion des figures (Si l'option est sélectionnée)
        if(self.options.groupe_figure == "combo"):
            while len(listeObjetsSelectionnes)>1:
                listeObjetsSelectionnes[0].attrib["d"] += " "+listeObjetsSelectionnes[-1].attrib["d"]
                listeObjetsSelectionnes.pop(-1)
                
        
        
        self.style = {'fill' : 'none', 'stroke' : self.options.couleur,'stroke-width' : str(self.options.epaisseur), 'stroke-linecap':'round'} # Style par defaut
        self.style_tirets_bronze = {'fill' : 'none', 'stroke' : self.options.couleur,'stroke-width' : str(self.options.epaisseur), 'stroke-linecap':'butt', 'stroke-dasharray':str(0.01*self.options.longueur_tiret_cuivre*self.options.periode)+","+str(0.01*self.options.longueur_espace_cuivre*self.options.periode)} # Style par defaut
        
        
        



        self.theta = -self.options.angle * math.pi/180 # Angle inclinaison hachure (par rapport à l'horizontale)
        
        #Liste des chemins finaux qui constitueront les hachures
        self.HACHURES_SORTIE = Path()
        self.HACHURES_SORTIE2 = Path()

        # Création du repère penché (DEBUG)-------------------------------
        self.ex = np.array([math.cos(self.theta),math.sin(self.theta)]) # Axe parallele aux hachures
        self.ey = np.array([-math.sin(self.theta),math.cos(self.theta)]) # Axe perpendiculare aux hachures
        self.repere = [self.ex,self.ey]
        if(self.DEBUG):
            layer.add(self.traceHachure(0,0,10*self.ex[0], 10*self.ex[1]))
            layer.add(self.traceHachure(0,0,10*self.ey[0], 10*self.ey[1]))


        # Pour chaque figure ,on fait les hachures
        for i in range(len(listeObjetsSelectionnes)): # Pour chaque objet sélectionné
        
            if(self.options.groupe_figure != "combo"): # Si objects différents, on recrée des hachures de zero à chaque fois
                self.HACHURES_SORTIE = Path()
                self.HACHURES_SORTIE2 = Path()
        
            cheminSelection = listeObjetsSelectionnes[i]#elementSelection.to_path_element()

            # On cherche la place qu'il prend (rectangle bounding box)
            Ymin,Ymax = self.getYminYmax(cheminSelection)
            Xmin,Xmax = self.getXminXmax(cheminSelection)
            
            # Prise en compte de l'offset
            Ymin += self.options.offset % self.options.periode
            Ymax += self.options.offset % self.options.periode

            objetCheminSelection = parse_path(cheminSelection.get_path())
            
            # DESSIN ACIER =========================================
            if(self.options.type_materiau == "acier"):
                for Y in np.arange(Ymin,Ymax+self.options.periode,self.options.periode):
                    P1 = self.ex*Xmin+self.ey*Y
                    P2 = self.ex*Xmax+self.ey*Y
                    listeIntersections = self.getIntersections(P1,P2,objetCheminSelection)
                    self.traceHachureEntreIntersections(self.HACHURES_SORTIE,listeIntersections)
                objetCheminHachure = PathElement.new(self.HACHURES_SORTIE)
                objetCheminHachure.style = self.style
                layer.add(objetCheminHachure)
            # DESSIN BRONZE ===========================================
            elif(self.options.type_materiau == "bronze"):
                for Y in np.arange(Ymin,Ymax+self.options.periode,self.options.periode):
                    # Trait plein
                    P1 = self.ex*Xmin+self.ey*Y
                    P2 = self.ex*Xmax+self.ey*Y
                    listeIntersections = self.getIntersections(P1,P2,objetCheminSelection)
                    self.traceHachureEntreIntersections(self.HACHURES_SORTIE,listeIntersections)
                    # Poitillés
                    P1 = self.ex*Xmin+self.ey*(Y+self.options.periode/2.)
                    P2 = self.ex*Xmax+self.ey*(Y+self.options.periode/2.)
                    listeIntersections = self.getIntersections(P1,P2,objetCheminSelection)
                    self.traceHachureEntreIntersections(self.HACHURES_SORTIE2,listeIntersections)
                objetCheminHachure = PathElement.new(self.HACHURES_SORTIE)
                objetCheminHachure.style = self.style
                layer.add(objetCheminHachure)
                objetCheminHachure2 = PathElement.new(self.HACHURES_SORTIE2)
                objetCheminHachure2.style = self.style_tirets_bronze
                layer.add(objetCheminHachure2)
            # DESSIN ALU ===========================================
            elif(self.options.type_materiau == "aluminium"):
                for Y in np.arange(Ymin,Ymax+self.options.periode,self.options.periode):
                    # Trait plein
                    P1 = self.ex*Xmin+self.ey*Y
                    P2 = self.ex*Xmax+self.ey*Y
                    listeIntersections = self.getIntersections(P1,P2,objetCheminSelection)
                    self.traceHachureEntreIntersections(self.HACHURES_SORTIE,listeIntersections)
                    # Poitillés
                    P1 = self.ex*Xmin+self.ey*(Y+self.options.periode*self.options.espace_aluminium/100.)
                    P2 = self.ex*Xmax+self.ey*(Y+self.options.periode*self.options.espace_aluminium/100.)
                    listeIntersections = self.getIntersections(P1,P2,objetCheminSelection)
                    self.traceHachureEntreIntersections(self.HACHURES_SORTIE,listeIntersections)
                objetCheminHachure = PathElement.new(self.HACHURES_SORTIE)
                objetCheminHachure.style = self.style
                layer.add(objetCheminHachure)
            # PLASTIQUE ===========================================
            elif(self.options.type_materiau == "plastique"):
                # Partie "horizontales"
                for Y in np.arange(Ymin,Ymax+self.options.periode,self.options.periode):
                    P1 = self.ex*Xmin+self.ey*Y
                    P2 = self.ex*Xmax+self.ey*Y
                    listeIntersections = self.getIntersections(P1,P2,objetCheminSelection)
                    self.traceHachureEntreIntersections(self.HACHURES_SORTIE,listeIntersections)
                # NOUVEAU THETA
                self.theta -= self.options.angle_plastique * math.pi/180
                # NOUVEAU REPERE
                self.ex = np.array([math.cos(self.theta),math.sin(self.theta)]) # Axe parallele aux hachures
                self.ey = np.array([-math.sin(self.theta),math.cos(self.theta)]) # Axe perpendiculare aux hachures
                # NOUVEAU BOUNDING BOX (reprend les lignes avant les ifs)
                Ymin,Ymax = self.getYminYmax(cheminSelection)
                Xmin,Xmax = self.getXminXmax(cheminSelection)
                Ymin += self.options.offset % self.options.periode
                Ymax += self.options.offset % self.options.periode
                for Y in np.arange(Ymin,Ymax+self.options.periode,self.options.periode):
                    P1 = self.ex*Xmin+self.ey*Y
                    P2 = self.ex*Xmax+self.ey*Y
                    listeIntersections = self.getIntersections(P1,P2,objetCheminSelection)
                    self.traceHachureEntreIntersections(self.HACHURES_SORTIE,listeIntersections)
                objetCheminHachure = PathElement.new(self.HACHURES_SORTIE)
                objetCheminHachure.style = self.style
                layer.add(objetCheminHachure)

        
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
                
        yMin = (yMin//self.options.periode)*self.options.periode # Décallage (pour être sûr de démarrer AVANT et finir APRES la figure)
        yMax = (yMax//self.options.periode+1)*self.options.periode
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
 
 
    def getIntersections(self,P1,P2,objetPath):
        intersections = []
        for troncon in objetPath: # Pour chaque troncon, selon ce que c'est (Ligne, Bezier, etc.)
            #debug("   - nouveau troncon !!!!!!!!!!"+str(troncon))
            if(type(troncon).__name__ in ["Line","Close","Arc", "QuadraticBezier"]):
                #debug("      - Ligne")
                PA = np.array([troncon.start.real, troncon.start.imag])
                PB = np.array([troncon.end.real, troncon.end.imag])
                if(self.seCroisentLignes(P1,P2,PA,PB)):
                    intersections.append(self.getIntersectionLignes(P1,P2,PA,PB))
                    #debug("         --> trouvé ")
            elif(type(troncon).__name__ == "CubicBezier"):
                #debug("      - Bezier")
                
                P_0 = np.array([troncon.start.real,troncon.start.imag])
                P_1 = np.array([troncon.control1.real,troncon.control1.imag])
                P_2 = np.array([troncon.control2.real,troncon.control2.imag])
                P_3 = np.array([troncon.end.real,troncon.end.imag])
                # Méthode 1
                intersections = intersections+self.getIntersectionsBezierCubic(P1,P2,P_0,P_1,P_2,P_3)
                    
        # On tri les intersections par ordre de distance à l'origine parallèlement aux hachures
        for i in range(len(intersections)-1):
            for j in range(len(intersections)-1):
                if(self.distanceALorigineParallelementHachure(intersections[j]) > self.distanceALorigineParallelementHachure(intersections[j+1])):
                    intersections[j],intersections[j+1] = intersections[j+1],intersections[j]
        return intersections
        
    def distanceALorigineParallelementHachure(self,P):
        return np.dot(self.ex,P)

                    
    
    def seCroisentLignes(self,P1,P2,PA,PB): # Ce qu'il y a près le "and" n'est peut être pas obligatoire (si les hachures prolongent à l'infini
        #P1 et P2 = extrémités max de la hachure.
        # PA et PB = extremités du segment

        def pseudocross2d(a, b):
            # Pseudo-produit vectoriel - scalaire 2D - compatible NumPy 2.x (avant on utilisait np.cross)
            return a[0]*b[1] - a[1]*b[0]
            
        if(pseudocross2d((P2-P1),(PA-PB))==0):
            return False
        return (pseudocross2d((P1-P2),(PA-P1))*pseudocross2d((P1-P2),(PB-P1))<0) #and (pseudocross2d((PB-PA),(P2-PA))*pseudocross2d((PB-PA),(P1-PA))<0)
      
    def getIntersectionLignes(self,P1,P2,PA,PB):
        # Résultat du système : P1C ^ P1P2 = 0 et PAC ^ PAPB = 0
        x1,y1,x2,y2 = P1[0],P1[1] , P2[0],P2[1]
        xA,yA,xB,yB = PA[0],PA[1] , PB[0],PB[1]
        
        if((yB-yA)*(x2-x1))-((y2-y1)*(xB-xA))==0: # Si parallèles
            return None
            
        yC = (-x1*(y2-y1)*(yB-yA)+y1*(x2-x1)*(yB-yA)+xA*(yB-yA)*(y2-y1)-yA*(xB-xA)*(y2-y1))/((x2-x1)*(yB-yA)-(xB-xA)*(y2-y1))
        xC = (x1*(y2-y1)*(xB-xA)-xA*(yB-yA)*(x2-x1)-y1*(x2-x1)*(xB-xA)+yA*(xB-xA)*(x2-x1))/(-(x2-x1)*(yB-yA)+(xB-xA)*(y2-y1))
        return np.array([xC,yC])
        
        
    # Fonction qui trace les hachures, une intersection sur deux
    def traceHachureEntreIntersections(self,cheminHachures,inter):
        if(len(inter)>1):
            for i in range(0,len(inter),2):
                if(i < len(inter)-1):
                    P1 = inter[i]
                    P2 = inter[i+1]
                    cheminHachures.append(ip.Move(P1[0],P1[1]))
                    cheminHachures.append(ip.Line(P2[0],P2[1]))
                    
                    
                    
                    
                    
    # Fonction qui renvoie la liste des points intersection entre la droite (P1,P2) et
    # la courbe de bezier cubique de points de controle P_i
    def getIntersectionsBezierCubic(self,P1,P2,P_0,P_1,P_2,P_3):
        #https://math.stackexchange.com/questions/2347733/intersections-between-a-cubic-b%C3%A9zier-curve-and-a-line
        intersec = []
    
        # Coef de l'equation de la droite (P1,P2)
        # equation de la droite : ax+by=d
        normale = np.array([[0,-1],[1,0]])@ np.transpose(P2-P1)
        normale = normale/np.linalg.norm(normale)
        a,b = normale
        d = np.dot(P1,normale)
                  
        # On cherche les racines de At³+Bt²+Ct+D=0
        A = -(a*P_0[0]+b*P_0[1]) + 3*(a*P_1[0]+b*P_1[1]) - 3*(a*P_2[0]+b*P_2[1]) + (a*P_3[0]+b*P_3[1])
        B = 3*(a*P_0[0]+b*P_0[1]) - 6 * (a*P_1[0]+b*P_1[1]) + 3*(a*P_2[0]+b*P_2[1])
        C = -3*(a*P_0[0]+b*P_0[1]) + 3*(a*P_1[0]+b*P_1[1])
        D = (a*P_0[0]+b*P_0[1]) - d
        
        #https://github.com/shril/CubicEquationSolver/blob/master/CubicEquationSolver.py
        # SI DEGRE 1
        if (A == 0 and B == 0):                     # Case for handling Liner Equation
            solutions = [(-D * 1.0) / C]                 # Returning linear root as numpy array.
        # SI DEGRE 2
        elif (A == 0):                              # Case for handling Quadratic Equations
            Delta = C**2 - 4.0 * B * D                       # Helper Temporary Variable
            if Delta > 0:
                x1 = (-C + sqrt(Delta)) / (2.0 * B)
                x2 = (-C - sqrt(Delta)) / (2.0 * B)
            #else: #Rien si solution double ou aucune

            solutions = [x1, x2]               # Returning Quadratic Roots as numpy array.
        else :
            f = ((3.0 * C / A) - ((B ** 2.0) / (A ** 2.0))) / 3.0
            g = (((2.0 * (B ** 3.0)) / (A ** 3.0)) - ((9.0 * B * C) / (A **2.0)) + (27.0 * D / A)) /27.0
            h = ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)
            
            if f == 0 and g == 0 and h == 0:            # All 3 Roots are Real and Equal
                #debug("Solution triple")
                # On a 3 racine identique, mais on en prend qu'une (c'est comme si la hachure rentrait, resortait et rerentrait au même point)
                if (D / A) >= 0:
                    solutions = [ (D / (1.0 * A)) ** (1 / 3.0) * -1 ]
                else:
                    solutions = [ (-D / (1.0 * A)) ** (1 / 3.0) ]
            elif h <= 0:                                # All 3 roots are Real
                #debug("3 Solutions reelles")
                i = math.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
                j = i ** (1 / 3.0)                      # Helper Temporary Variable
                k = acos(-(g / (2 * i)))                # Helper Temporary Variable
                L = j * -1                              # Helper Temporary Variable
                M = cos(k / 3.0)                   # Helper Temporary Variable
                N = sqrt(3) * sin(k / 3.0)    # Helper Temporary Variable
                P = (B / (3.0 * A)) * -1                # Helper Temporary Variable

                x1 = 2 * j * math.cos(k / 3.0) - (B / (3.0 * A))
                x2 = L * (M + N) + P
                x3 = L * (M - N) + P

                solutions = [x1, x2, x3]           # Returning Real Roots as numpy array.
            elif h > 0:                                 # One Real Root and two Complex Roots
                #debug("1 Solution reelle")
                R = -(g / 2.0) + math.sqrt(h)           # Helper Temporary Variable
                if R >= 0:
                    S = R ** (1 / 3.0)                  # Helper Temporary Variable
                else:
                    S = (-R) ** (1 / 3.0) * -1          # Helper Temporary Variable
                T = -(g / 2.0) - math.sqrt(h)
                if T >= 0:
                    U = (T ** (1 / 3.0))                # Helper Temporary Variable
                else:
                    U = ((-T) ** (1 / 3.0)) * -1        # Helper Temporary Variable

                x1 = (S + U) - (B / (3.0 * A))
                #debug(x1)
                #x2 = -(S + U) / 2 - (b / (3.0 * a)) + (S - U) * math.sqrt(3) * 0.5j
                #x3 = -(S + U) / 2 - (b / (3.0 * a)) - (S - U) * math.sqrt(3) * 0.5j

                solutions = [x1]           # Returning One Real Root and two Complex Roots as numpy array.

            for t in solutions:
                if(0<=t<=1):
                     intersec.append((1-t)**3*P_0 + 3*t*(1-t)**2*P_1 + 3*t**2*(1-t)*P_2 + t**3*P_3)  
                     
            return intersec
            
            
            
    def convertIntColor2Hex(self, i):
    	return "#"+("00000000"+hex(int(i))[2:])[-8:-2]
    	
    	
    	
    def updateCoordonneesRectangleDansLaBonneUnite(self, rect):
        # Récupérer les dimensions et positions du rectangle
        x = float(rect.get('x', '0'))  # Par défaut à 0 si absent
        y = float(rect.get('y', '0'))
        width = float(rect.get('width', '0'))
        height = float(rect.get('height', '0'))
        rx = float(rect.get('rx', '0'))  # Coins arrondis, par défaut 0
        ry = float(rect.get('ry', '0'))
        
        self.debug(x)
        self.debug(y)
        
        
        # Récupérer les unités actuelles du document
        unit = self.svg.unit  # Exemple : 'mm', 'px', etc.
        self.debug(unit)
        scale_factor = self.svg.uutounit(1, "px")  # Conversion de 1 px vers l'unité actuelle
        self.debug("Facteur = "+str(scale_factor))
        
        # Convertir les dimensions en pixels (ou dans l'unité du document)
        x /= scale_factor #self.svg.unittouu(x)
        y /= scale_factor # self.svg.unittouu(y)
        width  /= scale_factor # self.svg.unittouu(width)
        height /= scale_factor # self.svg.unittouu(height)
        rx /= scale_factor # self.svg.unittouu(rx)
        ry /= scale_factor # self.svg.unittouu(ry)
        
        self.debug(x)
        self.debug(y)
        
        #rect.set('x',str(x))
        #rect.set('y',str(y))
        

        
if __name__ == '__main__':
    Hachures().run()
