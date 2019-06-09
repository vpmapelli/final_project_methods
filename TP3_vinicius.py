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
        self.value = np.array(value)
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
        K_lhs = sparse.diags([2*np.ones(2*n_gdl),1*np.ones(n_gdl)],[0,n_gdl],(2*n_gdl,2*n_gdl),"coo")
        K_lhs = (self.length/6)*K_lhs
        
        
        #Referente ao primeiro termo do lado direito da equacao variacional do livro de Donea
        diag = np.ones(n_gdl)
        K_convective = sparse.diags([np.append(-diag,diag),-diag,diag],[0,n_gdl,-n_gdl],(2*n_gdl,2*n_gdl),"coo")
        K_convective = 0.5*K_convective
        
        #Referente ao segundo termo do lado direito da equacao variacional do livro de Donea
        #Matriz simetrica = salvo apenas os valores da parte triangular superior
        diag = np.ones(2*n_gdl)
        K_diffusive = sparse.spdiags([diag,-diag],[0,n_gdl],2*n_gdl,2*n_gdl,"coo")
        K_diffusive = K_diffusive/(self.length)
        
        
        #LEMBRAR QUE K_lhs,K_diffusive ESTA SALVA PELA METADE (MATRIZ SIMETRICA!)!!!!!!!!!!
        return K_lhs,K_convective,K_diffusive
    
class cls_problem():
    def __init__(self, node, elements, bc, time_increment):
        self.gdl        = node[0].gdl
        self.Nnodes     = len(node)
        self.Nelem      = len(elements)
        self.elements   = elements
        self.gdl        = gdl
        self.bc         = bc
        self.dt         = time_increment
        
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
    
    def problem_solve(self,U):
        
        
        
        '''Sabendo que foi utilizada um espacamento homogeneo e nao ha dependencia temporal das matrizes
        dessa forma, podemos calcular as matrizes de um elemento apenas uma vez'''
        gamma = self.elements[0].material.gamma
        
        Ke_lhs,Ke_convective,Ke_diffusive = self.elements[0].element_matrices()
        gdl = self.gdl
        dt = self.dt
        
        rows = []
        cols = []
        rows_c = []
        cols_c = []
        data_lhs = []
        data_convective = []
        data_diffusive = []
        
        rows_A = []
        cols_A = []
        data_A = []
        
        for k,item in enumerate(self.elements):
            #Matrizes Ke_lhs e Ke_convective sao simetricas e possuem valores nas mesmas posicoes
            rows = np.append(rows, Ke_lhs.row + k*gdl)
            cols = np.append(cols, Ke_lhs.col  + k*gdl)
            
            #Matriz Ke_convective nao e anti-simetrica
            rows_c = np.append(rows_c, Ke_convective.row + k*gdl)
            cols_c = np.append(cols_c, Ke_convective.col + k*gdl)
            
            data_lhs        = np.append(data_lhs,        Ke_lhs.data)
            data_convective = np.append(data_convective, Ke_convective.data)
            data_diffusive  = np.append(data_diffusive,  Ke_diffusive.data)
            
            #Construindo a matriz A de forma a ser aplicada no vetor U
            rows_A = np.append(rows_A, np.array([0,1,1,1,2,2,2]) + k*gdl)
            cols_A = np.append(cols_A, np.array([1,0,1,2,0,1,2]) + k*gdl)
            

            
            v = U[k*gdl+1]/U[k*gdl]
            E = U[k*gdl+2]/U[k*gdl]
            data_A = np.append(data_A, np.array([1,
                                                 -0.5*(3-gamma)*(v**2.0),
                                                 (3-gamma)*(v),
                                                 gamma-1,
                                                 (gamma-1)*(v)**3.0-gamma*E*v,
                                                 gamma*E-1.5*(gamma-1)*(v**2.0),
                                                 gamma*v]))
            

        
        '''A matriz A é aplicada sobre os nós. O último (for loop) corre os elementos,
        e foi utlizado para evitar utilizar um segundo (for loop) correndo os nos. No entando,
        como Nelementos = Nnodes -1, a aplicacao de A no ulitmo node deve ser feita após o loop'''
        N = self.Nnodes
        rows_A = np.append(rows_A, np.array([0,1,1,1,2,2,2]) + (N-1)*gdl)
        cols_A = np.append(cols_A, np.array([1,0,1,2,0,1,2]) + (N-1)*gdl)
        
        v = U[(N-1)*gdl+1]/U[(N-1)*gdl]
        E = U[(N-1)*gdl+2]/U[(N-1)*gdl]
        data_A = np.append(data_A, np.array([1,
                                             -0.5*(3-gamma)*(v**2.0),
                                             (3-gamma)*(v),
                                             gamma-1,
                                             (gamma-1)*(v)**3.0-gamma*E*v,
                                             gamma*E-1.5*(gamma-1)*(v**2.0),
                                             gamma*v]))
    

    
    
        K_global = sparse.csr_matrix((data_lhs, (rows, cols)), shape=(gdl*N,gdl*N))
        K_global = K_global + K_global.T - sparse.diags(K_global.diagonal(), dtype='float')
        
        K_c = sparse.csr_matrix((data_convective, (rows_c, cols_c)), shape=(gdl*N,gdl*N))
        K_c.eliminate_zeros() #eliminar os valores iguais a zero
        
        K_d = sparse.csr_matrix((data_diffusive, (rows, cols)), shape=(gdl*N,gdl*N))
        K_d = K_d + K_d.T - sparse.diags(K_d.diagonal(), dtype='float')
        
        A = sparse.csr_matrix((data_A, (rows_A, cols_A)), shape=(gdl*N,gdl*N))
        
        #Construindo o vetor livre
        F = A.dot(U)
        A_sq_U = A.dot(F)
        
        f_global = dt*K_c.dot(F) -0.5*dt*dt*K_d.dot(A_sq_U)
        
        #Boundary conditions
        w = 1e20
        for k in range(0,len(bc)):
            pos = bc[k].node.id
            
#            if pos==0:
#                f_global[pos*gdl:(pos+1)*gdl] = f_global[pos*gdl:(pos+1)*gdl] + dt*bc[k].value    
#            elif pos==N-1:
#                f_global[pos*gdl:(pos+1)*gdl] = f_global[pos*gdl:(pos+1)*gdl] - dt*bc[k].value    
        
            K_global[pos*gdl,pos*gdl]     =  K_global[pos*gdl,pos*gdl] + w
            K_global[pos*gdl+1,pos*gdl+1] =  K_global[pos*gdl+1,pos*gdl+1] + w
            K_global[pos*gdl+2,pos*gdl+2] =  K_global[pos*gdl+2,pos*gdl+2] + w
            
            f_global[pos*gdl] = bc[k].value[0]*w
            f_global[pos*gdl+1] = bc[k].value[1]*w
            f_global[pos*gdl+2] = bc[k].value[2]*w
            
        return spsolve(K_global, f_global)
              

######################################################################################
##Programa principal
        
#Parâmetros do problema
N_final = 101       #Numero de elementos
gamma = 1.4         #Coeficiente de expansao adiabatica
delta_t = 1.5e-3    #Incremento temporal
gdl = 3             #Graus de liberdade



#Criando nos
node = []
for k in range(0,N_final):
    node.append(cls_node(k*1/(N_final-1),k,gdl))

#Criando elementos
element = []    
for k in range(0,N_final-1):
        material = cls_material(gamma)
        element.append(cls_element_1D((node[k], node[k+1]), material))

#Aplicando condicoes de contorno
bc = []
#bc.append(cls_boundary_conditions(node[0]      ,   gdl, [0,  1,0], 'flux'))
#bc.append(cls_boundary_conditions(node[N_final-1], gdl, [0,0.1,0], 'flux'))
bc.append(cls_boundary_conditions(node[0]      ,   gdl, [0,0,0], 'flux'))
bc.append(cls_boundary_conditions(node[N_final-1], gdl, [0,0,0], 'flux'))


#Definindo o problema
problem = cls_problem(node,element,bc,delta_t)


#Condicao inicial
U = np.zeros(gdl*N_final)
for k in range(0,N_final):
    if node[k].coords <= 0.5:
        U[k*gdl:(k+1)*gdl] = np.array([1,0,2.5])
    else:
        U[k*gdl:(k+1)*gdl] = np.array([0.125, 0, 0.25])
            

#Plotando condicao inicial
plt.close('all')

plt.figure(1)
x = np.linspace(0,1,N_final)

plt.subplot(2,2,1)
plt.title('Density')
plt.plot(x,U[0:N_final*gdl:gdl])

plt.subplot(2,2,2)
plt.title('Velocity')
v = U[1:N_final*gdl+1:gdl]/U[0:N_final*gdl:gdl]
plt.plot(x,v)

plt.subplot(2,2,3)
plt.title('Pressure')
P = (gamma-1)*(U[2:N_final*gdl+2:gdl] - U[1:N_final*gdl+1:gdl]*v/2)
plt.plot(x,P)

plt.subplot(2,2,4)
plt.title('Energy')
E = U[2:N_final*gdl+2:gdl]
plt.plot(x,E)

##Main loop
t = 0
t_final = 0.2
densities = U[0:N_final*gdl:gdl].copy()

time = []
time.append(t)

while t<t_final:
    Delta_U = problem.problem_solve(U)
    U += Delta_U
    t += delta_t
    
    densities = np.vstack([densities,U[0:N_final*gdl:gdl]])
    time.append(t)
    

#Ler dados do resultado analitico
analytic_data = open('analytic.txt','r').read()
lines = analytic_data.split('\n')
rho_analy   = np.array(lines[0].split(','),dtype=float)
u_analy     = np.array(lines[1].split(','),dtype=float)
P_analy     = np.array(lines[2].split(','),dtype=float)
rhoE_analy  = np.array(lines[3].split(','),dtype=float)
x_analy     = np.linspace(0,1,len(rho_analy))

#Plot em t=t_final
plt.figure(2,[10,10])

plt.subplot(2,2,1)
plt.title('Density')
plt.grid(1,'major','both')
plt.plot(x,U[0:N_final*gdl:gdl],'k+')
plt.plot(x_analy,rho_analy,'k')

plt.subplot(2,2,2)
plt.title('Velocity')
plt.grid(1,'major','both')
v = U[1:N_final*gdl+1:gdl]/U[0:N_final*gdl:gdl]
plt.plot(x,v,'k+')
plt.plot(x_analy,u_analy,'k')

plt.subplot(2,2,3)
plt.title('Pressure')
plt.grid(1,'major','both')
P = (gamma-1)*(U[2:N_final*gdl+2:gdl] - U[1:N_final*gdl+1:gdl]*v/2)
plt.plot(x,P,'k+')
plt.plot(x_analy,P_analy,'k')

plt.subplot(2,2,4)
plt.title('Energy')
plt.grid(1,'major','both')
E = U[2:N_final*gdl+2:gdl]
plt.plot(x,E,'k+')
plt.plot(x_analy,rhoE_analy,'k')

#Contour plot da densidade
X,T = np.meshgrid(x,np.array(time))

plt.figure(3)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
plt.contourf(X,T,densities,100)
plt.ylabel('Tempo [s]')
plt.xlabel('x')
plt.title(r'$\rho$ ' 'vs t')
plt.colorbar()
