'''
Aluno: Vinicius Pessoa Mapelli N USP 7593457

Solucao numerica de Sod Shock Tube utilizando elementos finitos, como elementos lineares ou quadraticos

Trabalho final desenvolvido para disciplina de SEM5738 Metodos Numéricos, 
ministrada pelo Prof. Dr. Ricardo no primeiro semestre de 2019.'''

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
        self.bc_type = bc_type

class cls_element_1D():
    def __init__(self, nodes, material):
        self.nodes = nodes
        self.material = material
        self.centroid = self.compute_centroid()
        self.length = self.compute_length()

        
    def compute_centroid(self):
        if len(self.nodes) == 2:
            return (self.nodes[1].coords+self.nodes[0].coords) *0.5
        elif len(self.nodes) == 3:
            return self.nodes[1]
            
    def compute_length(self):
        if len(self.nodes) == 2:
            return np.sqrt(np.sum((self.nodes[1].coords - self.nodes[0].coords)**2.0))    
        elif len(self.nodes) == 3:
            return np.sqrt(np.sum((self.nodes[2].coords - self.nodes[0].coords)**2.0))  
        
    def element_matrices(self):
        
        n_gdl = self.nodes[0].gdl
            
        if (len(self.nodes)==2):
            #Matriz do sistema que multiplica (U^(n+1) - U(n))
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
                
        elif (len(self.nodes)==3):
            #Matriz do sistema que multiplica (U^(n+1) - U(n))
            #Matriz simetrica = necessario salvar apenas os valores da parte superior
            K_lhs = sparse.diags([np.concatenate([4*np.ones(n_gdl),16*np.ones(n_gdl),4*np.ones(n_gdl)]),2*np.ones(2*n_gdl),-1*np.ones(n_gdl)],[0,n_gdl,2*n_gdl],(3*n_gdl,3*n_gdl),"coo")
            K_lhs = (self.length/30)*K_lhs
            
            
            #Referente ao primeiro termo do lado direito da equacao variacional do livro de Donea
            diag1 = -4*np.ones(2*n_gdl)
            diag2 = np.ones(n_gdl)
            K_convective = sparse.diags([np.concatenate([-3*np.ones(n_gdl),np.zeros(n_gdl),3*np.ones(n_gdl)]),diag1,diag2,-diag1,-diag2],[0,n_gdl,2*n_gdl,-n_gdl,-2*n_gdl],(3*n_gdl,3*n_gdl),"coo")
            K_convective = K_convective/6.0
            
            #Referente ao segundo termo do lado direito da equacao variacional do livro de Donea
            #Matriz simetrica = salvo apenas os valores da parte triangular superior
            K_diffusive = sparse.diags([np.concatenate([7*np.ones(n_gdl),16*np.ones(n_gdl),7*np.ones(n_gdl)]),-8*np.ones(2*n_gdl),1*np.ones(n_gdl)],[0,n_gdl,2*n_gdl],(3*n_gdl,3*n_gdl),"coo")
            K_diffusive = K_diffusive/(3*self.length)
        
        
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

    def problem_solve_linear(self,U):
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
              

    def problem_solve_quadratic(self,U):
        '''Sabendo que foi utilizada um espacamento homogeneo e nao ha dependencia temporal das matrizes
        dessa forma, podemos calcular as matrizes de um elemento apenas uma vez'''
        gamma = self.elements[0].material.gamma
        N = self.Nnodes
        
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
        

        for k,item in enumerate(self.elements):
            #Matrizes Ke_lhs e Ke_convective sao simetricas e possuem valores nas mesmas posicoes
            rows = np.append(rows, Ke_lhs.row  + k*2*gdl)
            cols = np.append(cols, Ke_lhs.col  + k*2*gdl)
            
            #Matriz Ke_convective nao e anti-simetrica
            rows_c = np.append(rows_c, Ke_convective.row + k*2*gdl)
            cols_c = np.append(cols_c, Ke_convective.col + k*2*gdl)
            
            data_lhs        = np.append(data_lhs,        Ke_lhs.data)
            data_convective = np.append(data_convective, Ke_convective.data)
            data_diffusive  = np.append(data_diffusive,  Ke_diffusive.data)
        

        #Construindo a matriz A de forma a ser aplicada no vetor U
        rows_A = []
        cols_A = []
        data_A = []
                
        for k in range(0,N):
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
    
################################################################################################
################################### PROGRAMA PRINCIPAL #########################################
plt.close('all')
#Parâmetros do problema
element_type   = 'linear'  #'linear' ou 'quadratic' (o programa aceita apenas elementos de um tipo)
N_final        = 101       #Numero de nodes (para utilizar elementos quadraticos, numero de nodes deve ser impar maior que 3) 
gamma          = 1.4       #Coeficiente de expansao adiabatica
delta_t        = 1.5e-3    #Incremento temporal [s]
t_final        = 0.2       #Tempo final de simulacao [s]
gdl            = 3         #Graus de liberdade


###################################  CRIANDO OS NOS  ###########################################
node = []
for k in range(0,N_final):
    node.append(cls_node(k*1/(N_final-1),k,gdl))

################################### CRIANDO ELEMENTOS ##########################################
element = []
if element_type == 'linear':
    for k in range(0,N_final-1):
            material = cls_material(gamma)
            element.append(cls_element_1D((node[k], node[k+1]), material))       
elif (element_type == 'quadratic'):
    for k in range(0,N_final-1,2):
        material = cls_material(gamma)
        element.append(cls_element_1D((node[k], node[k+1], node[k+2]), material))
else:
    raise Exception('Tipo de elemento invalido')
    

################################### APLICANDO C.Cs ###########################################
bc = []
#bc.append(cls_boundary_conditions(node[0]      ,   gdl, [0,  1,0], 'flux'))
#bc.append(cls_boundary_conditions(node[N_final-1], gdl, [0,0.1,0], 'flux'))
bc.append(cls_boundary_conditions(node[0]      ,   gdl, [0,0,0], 'value'))
bc.append(cls_boundary_conditions(node[N_final-1], gdl, [0,0,0], 'value'))


################################### APLICANDO C.I, ###########################################
U = np.zeros(gdl*N_final)
for k in range(0,N_final):
    if node[k].coords <= 0.5:
        U[k*gdl:(k+1)*gdl] = np.array([1,0,2.5])
    else:
        U[k*gdl:(k+1)*gdl] = np.array([0.125, 0, 0.25])

################################### PLOTANDO A C.I. ##########################################
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

################################### DEFININDO PROBLEMA #######################################
problem = cls_problem(node,element,bc,delta_t)


################################### LOOP PRINCIPAL  ###########################################
densities = U[0:N_final*gdl:gdl].copy() #Criando contourf plot da densidade em x,t


t = 0
time = []
time.append(t)

if element_type == 'linear':
    while t<t_final:
        Delta_U = problem.problem_solve_linear(U)
        U += Delta_U
        t += delta_t
        
        densities = np.vstack([densities,U[0:N_final*gdl:gdl]]) #Criando contourf plot da densidade em x,t
        time.append(t)
else:
    while t<t_final:
        Delta_U = problem.problem_solve_quadratic(U)
        U += Delta_U
        t += delta_t
        
        densities = np.vstack([densities,U[0:N_final*gdl:gdl]]) #Criando contourf plot da densidade em x,t
        time.append(t)
    
    
############################# LENDO DADOS DA SOLUCAO ANALITICA ##################################
analytic_data = open('analytic.txt','r').read()
lines = analytic_data.split('\n')
rho_analy   = np.array(lines[0].split(','),dtype=float)
u_analy     = np.array(lines[1].split(','),dtype=float)
P_analy     = np.array(lines[2].split(','),dtype=float)
rhoE_analy  = np.array(lines[3].split(','),dtype=float)
x_analy     = np.linspace(0,1,len(rho_analy))

############################# PLOTANDO RESULTADOS DO FEM EM T=T_FINAL ###########################

#Alguns dados de controle do plot
fem_style = 'k+'        #estilo da linha do resultado pelo fem
x_axis = (0,1)          #limites do eixo x (plot original usa intervalo pouco maior que 0 e 1)
analytic_plot = 'on'    #'on' 'off' (mostrar a solucao analitica)
figure_size = [10,10]   #tamanho da figura (polegadas)

plt.figure(2,figure_size)

x = np.linspace(0,1,N_final)
plt.subplot(2,2,1)
plt.title('Density')
plt.grid(1,'major','both')
plt.plot(x,U[0:N_final*gdl:gdl],fem_style)
plt.xlim(x_axis)
if analytic_plot == 'on': plt.plot(x_analy,rho_analy,'k')


plt.subplot(2,2,2)
plt.title('Velocity')
plt.grid(1,'major','both')
v = U[1:N_final*gdl+1:gdl]/U[0:N_final*gdl:gdl]
plt.plot(x,v,fem_style)
plt.xlim(x_axis)
if analytic_plot == 'on': plt.plot(x_analy,u_analy,'k')

plt.subplot(2,2,3)
plt.title('Pressure')
plt.grid(1,'major','both')
P = (gamma-1)*(U[2:N_final*gdl+2:gdl] - U[1:N_final*gdl+1:gdl]*v/2)
plt.plot(x,P,fem_style)
plt.xlim(x_axis)
if analytic_plot == 'on': plt.plot(x_analy,P_analy,'k')

plt.subplot(2,2,4)
plt.title('Energy')
plt.grid(1,'major','both')
E = U[2:N_final*gdl+2:gdl]
plt.plot(x,E,fem_style)
plt.xlim(x_axis)
if analytic_plot == 'on': plt.plot(x_analy,rhoE_analy,'k')

plt.savefig('fem_sod_shock_tube.eps', format='eps', dpi=1000)

############################# DENSIDADE EM X,T ###########################
X,T = np.meshgrid(x,np.array(time))

plt.figure(3)
plt.contourf(X,T,densities,100)
plt.ylabel('Tempo [s]')
plt.xlabel('x')
plt.title(r'$\rho$ ' 'vs t')
plt.colorbar()

plt.savefig('rho_xt,eps', format='eps', dpi=1000)
