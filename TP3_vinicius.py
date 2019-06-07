# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse, linalg
from scipy.sparse.linalg import spsolve
import time as sys_time
#from sympy.geometry import Point, Segment

class cls_node():
    def __init__(self,coords, node_id, gdl_node):
        self.coords = np.array(coords)
        self.id = node_id
        self.gdl = gdl_node
        
    def set_coords(self,coords):
        self.coords = np.array(coords)
    
class cls_material():
    def __init__(self, adiabatic_expansion):
        self.gamma               = adiabatic_expansion        
        
class cls_boundary_conditions():
    def __init__(self, node, gdl, value, bc_type):
        self.node = node
        self.gdl = gdl
        self.value = value
#        self.bc = bc
        self.bc_type = bc_type

class cls_element_1D():
    def __init__(self, nodes, material):
        self.nodes = nodes
        self.material = material
        self.centroid = self.compute_centroid()
        self.length = self.compute_length()

        
    def compute_centroid(self):
        return (self.nodes[1].coords+self.nodes[0].coords) *0.5
    
    def compute_length(self):
        return np.sqrt(np.sum((self.nodes[1].coords - self.nodes[0].coords)**2.0))
        
    def convective_term(self,n_gdl):
        #Referente ao primeiro termo do lado direito da equacao variacional do livro de Donea
        diag = np.ones(n_gdl)
        A = sparse.diags([np.append(-diag,diag),-diag,diag],[0,n_gdl,-n_gdl],(2*n_gdl,2*n_gdl),"csr")
        A = 0.5*A
        #Retorna a matriz do elemento, que deve ser multiplicada pelos valores dos nos, conhecidos previamente do passo anterior
        return A
    
    def diffusive_term(self,n_gdl):
        #Referente ao segundo termo do lado direito da equacao variacional do livro de Donea
        diag = np.ones(2*n_gdl)
        A = sparse.spdiags([diag,-diag,-diag],[0,n_gdl,-n_gdl],2*n_gdl,2*n_gdl,"csr")
        A = A/(self.length)
        return A
        
    def element_matrices(self):
        
        #Matriz do sistema que multiplica (U^(n+1) - U(n))
        n_gdl = self.nodes[0].gdl
        
        #Matriz simetrica = necessario salvar apenas os valores da parte superior
        K_lhs = sparse.diags([2*np.ones(2*n_gdl),1*np.ones(n_gdl)],[0,n_gdl],(2*n_gdl,2*n_gdl),"csr")
        K_lhs = (self.length/6)*K_lhs
        
        
        #Referente ao primeiro termo do lado direito da equacao variacional do livro de Donea
        diag = np.ones(n_gdl)
        K_convective = sparse.diags([np.append(-diag,diag),-diag,diag],[0,n_gdl,-n_gdl],(2*n_gdl,2*n_gdl),"csr")
        K_convective = 0.5*K_convective
        
        #Referente ao segundo termo do lado direito da equacao variacional do livro de Donea
        diag = np.ones(2*n_gdl)
        K_diffusive = sparse.spdiags([diag,-diag,-diag],[0,n_gdl,-n_gdl],2*n_gdl,2*n_gdl,"csr")
        K_diffusive = K_diffusive/(self.length)
        
        
        #LEMBRAR QUE K_LHS ESTA SALVA PELA METADE (MATRIZ SIMETRICA!)!!!!!!!!!!
        return K_lhs,K_convective,K_diffusive
    
class cls_problem():
    def __init__(self, node, elements, gdl, bc, time_increment):
        self.gdl        = gdl
        self.Nnodes     = len(node)
        self.Nelem      = len(elements)
        self.elements   = elements
        self.gdl        = gdl
        self.bc         = bc
        self.dt         = time_increment
        self.U          = np.zeros(self.Nnodes*gdl)
        
    #Recuperar o valor de U de acordo com a coordenada
    def eval_U(self,node_id):
        #Como o problema e 1D, tomaremos vantagem dessa caracteristica e substituiremos a coordenada pelo node_id
        #Uma vez que e bastante facil recuperar a coordenada dado o node_id (e vice-versa)
        
        return self.U[gdl*node_id:(gdl*node_id+self.gdl)]
    
    def eval_F(self,node_id):
        #Sabendo que o gamma eh constante, podemos pegar de qualquer elemento
        gamma = self.elements[0].material.gamma
        
        #Calculo do F ja especifico para o problema de Shock-Tube a ser resolvido
        F = np.zeros(self.gdl)
        
        v = self.U[gdl*node_id+1]/self.U[gdl*node_id]
        P = (gamma-1)*(self.U[gdl*node_id+2] - self.U[gdl*node_id+1]*v/2)
        
        F[0] = self.U[gdl*node_id+1]
        F[1] = self.U[gdl*node_id+1]*v + P
        F[2] = v*(self.U[gdl*node_id+2] + P)
        
        return F
        

#LEMBRAR QUE K_LHS ESTA SALVA PELA METADE (MATRIZ SIMETRICA!)!!!!!!!!!!
        
    
    
    
#    def problem_solve(self):
#        Nnodes  = len(node)
#        Nelem   = len(element)
#        
#    #    K_global = np.zeros((Nnodes,Nnodes))
#        f_global = np.zeros(Nnodes)
#        u = np.zeros(Nnodes)
#        
#        rows = []
#        cols = []
#        data = []
#        for k,item in enumerate(element):
#            K_elem, f_elem = item.element_matrices()
#            rows.append(K_elem[0])
#            cols.append(K_elem[1])
#            data.append(K_elem[2])
#            
#            f_global[k] += f_elem[0]
#            f_global[k+1] += f_elem[1]
#            
#        rows = np.array(rows, dtype='int').flatten()
#        cols = np.array(cols, dtype='int').flatten()
#        data = np.array(data, dtype='float').flatten()
#            
#        K_global = sparse.csr_matrix((data, (rows, cols)), shape=(Nnodes,Nnodes))
#        K_global = K_global + K_global.T - sparse.diags(K_global.diagonal(), dtype='float')
#            
#        # Boundary conditions
#        w = 1e20
#        
#        for k in range(0,len(bc)):
#            pos = bc[k].node.id - 1
#            K_global[pos,pos] = K_global[pos,pos] + w
#            f_global[pos] = bc[k].value*w
#    
#        u = spsolve(K_global, f_global)
#        
#        return u        





node = []

#Criando nos
N_final = 1
#for k in range(0,N_final):
#    node.append(cls_node(k*1/(N_final-1),k,3))

gdl = 3
node.append(cls_node(0,0,gdl))
node.append(cls_node(2,1,gdl))


#Criando elementos
element = []    
for k in range(0,N_final-1):
        material = cls_material(1.4)
        element.append(cls_element_1D((node[k], node[k+1]), material))

gamma = 1.4
delta_t = 1    
material = cls_material(gamma)
element.append(cls_element_1D((node[0], node[1]), material))
K_lhs,K_convective,K_diffusive = element[0].element_matrices()

print(K_lhs.todense())       
print(K_convective.todense())
print(K_diffusive.todense())

problem = cls_problem(node,element,gdl,0,delta_t)
problem.U = np.append(np.array([1,0,2.5]),np.array([0.125,0,0.25]))
    
#Aplicando condicoes de contorno
#bc = []
#bc.append(cls_boundary_conditions(node[0]      , 1, 0.0, 'U'))
#bc.append(cls_boundary_conditions(node[N_final-1], 1, 0.0, 'U'))
    
