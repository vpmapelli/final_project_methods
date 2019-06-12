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
        
        #caso de elementos lineares    
        if (len(self.nodes)==2): 
            ####################### MATRIZ M ###############################################
            #Matriz do sistema que multiplica (U^(n+1) - U(n)) (Mass matrix)
            #M = h/6*[[2I,1I],[1I,2I]] (em que I é a identidade 3x3)            
            #Matriz simetrica = necessario salvar apenas os valores da parte superior
            M = sparse.coo_matrix(np.array([[2,1],[0,2]])) #M[1,0] = 0 pois iremos guardar apenas os valores da parte triangular superior
            M = sparse.kron(M,sparse.eye(n_gdl),"coo") #Produto kron, nesse caso, multiplica a matriz identidade de dimensao gdl x gdl pelo valor de cada uma das posicoes do primeiro argumento M 
            M = (self.length/6)*M
            

            ####################### MATRIZ C ###############################################
            #Referente ao primeiro termo do lado direito da equacao variacional do livro de Donea
            #C = 1/2*[[-1I,-1I],[2I,2I]] (em que I é a identidade 3x3)
            C = sparse.coo_matrix(np.array([[-1,-1],[1,1]]))
            C = sparse.kron(C,sparse.eye(n_gdl),"coo") #Produto kron, nesse caso, multiplica a matriz identidade de dimensao gdl x gdl pelo valor de cada uma das posicoes do primeiro argumento M 
            C = 0.5*C
            
            
            ####################### MATRIZ K ###############################################
            #Referente ao segundo termo do lado direito da equacao variacional do livro de Donea
            #Matriz simetrica = necessario salvar apenas os valores da parte superior
            #K = (1/h)*[[1I,-1I],[1I,-1I]] (em que I é a identidade 3x3)   
            K = sparse.coo_matrix(np.array([[1,-1],[0,1]])) #M[1,0] = 0 pois iremos guardar apenas os valores da parte triangular superior
            K = sparse.kron(K,sparse.eye(n_gdl),"coo") #Produto kron, nesse caso, multiplica a matriz identidade de dimensao gdl x gdl pelo valor de cada uma das posicoes do primeiro argumento M 
            K = K/(self.length)        
            
        #caso de elementos quadraticos
        elif (len(self.nodes)==3):
            ####################### MATRIZ M ###############################################
            #Matriz do sistema que multiplica (U^(n+1) - U(n)) (Mass matrix)
            #M = L/30*[[4I,2I,-1I],[2I,16I,2I],[-1I,2I,4I]] (em que I é a identidade 3x3)            
            #Matriz simetrica = necessario salvar apenas os valores da parte superior
            M = sparse.coo_matrix(np.array([[4,2,-1],[0,16,2],[0,0,4]])) #Valores da parte triangular inferior zerados, pois matriz simetrica
            M = sparse.kron(M,sparse.eye(n_gdl),"coo") #Produto kron, nesse caso, multiplica a matriz identidade de dimensao gdl x gdl pelo valor de cada uma das posicoes do primeiro argumento M 
            M = (self.length/30)*M
            

            ####################### MATRIZ C ###############################################
            #Referente ao primeiro termo do lado direito da equacao variacional do livro de Donea
            #C = 1/6*[[-3I,-4I,1I],[4I,0,4I],[-1I,4I,3I]] (em que I é a identidade 3x3)
            C = sparse.coo_matrix(np.array([[-3,-4,1],[4,0,-4],[-1,4,3]]))
            C = sparse.kron(C,sparse.eye(n_gdl),"coo") #Produto kron, nesse caso, multiplica a matriz identidade de dimensao gdl x gdl pelo valor de cada uma das posicoes do primeiro argumento M 
            C = (1/6)*C
            
            
            ####################### MATRIZ K ###############################################
            #Referente ao segundo termo do lado direito da equacao variacional do livro de Donea
            #Matriz simetrica = necessario salvar apenas os valores da parte superior
            #K = (1/(3L))*[[7I,-8I,1I],[-8I,16I,-8I],[1I,-8I,7I]] (em que I é a identidade 3x3)   
            K = sparse.coo_matrix(np.array([[7,-8,1],[0,16,-8],[0,0,7]])) #Valores da parte triangular inferior zerados, pois matriz simetrica
            K = sparse.kron(K,sparse.eye(n_gdl),"coo") #Produto kron, nesse caso, multiplica a matriz identidade de dimensao gdl x gdl pelo valor de cada uma das posicoes do primeiro argumento M 
            K = K/(3*self.length)        
        
        return M,C,K
    
class cls_problem():
    def __init__(self, node, elements, bc, time_increment):
        self.gdl        = node[0].gdl
        self.Nnodes     = len(node)
        self.Nelem      = len(elements)
        self.elements   = elements
        self.gdl        = gdl
        self.bc         = bc
        self.dt         = time_increment

    def compute_global_matrices(self):
        gdl = self.gdl
        
        rows = []
        cols = []
        rows_c = []
        cols_c = []
        data_M = []
        data_C = []
        data_K = []
                
        positioner = 0 #variavel auxiliar da posicao do elemento (0,0) na matriz global
        for k,item in enumerate(self.elements):
            M_elem,C_elem,K_elem = item.element_matrices()

            #Matrizes M_elem e K_elem sao simetricas e possuem valores nas mesmas posicoes
            #Para cada iteracao, deslocamos o valor em positioner nas colunas e linhas na matriz global
            rows = np.append(rows, M_elem.row + positioner)
            cols = np.append(cols, M_elem.col + positioner)
            
            #Matriz C_elem nao e simetrica (mais valores foram salvos)
            rows_c = np.append(rows_c, C_elem.row + positioner)
            cols_c = np.append(cols_c, C_elem.col + positioner)
            
            '''Se existirem valores na mesma posicao, como no caso nas posicoes que mais de um elemento possuem valor nao nulo,
            eles serao automaticamente somados na construcao da matriz esparsa'''
            data_M = np.append(data_M, M_elem.data)
            data_C = np.append(data_C, C_elem.data)
            data_K = np.append(data_K, K_elem.data)
            
            #A cada elemento, deslocamos gdl*(N_elem-1) linhas e colunas na matriz global
            #Se o elemento for linear, gdl*(N_elem-1) = 3
            #Se o elemento for linear, gdl*(N_elem-1) = 6
            N_elem = len(item.nodes)
            positioner += gdl*(N_elem-1)
            
        #Montando matrizes globais
        N  = self.Nnodes
        K_global = sparse.csr_matrix((data_K, (rows, cols)), shape=(gdl*N,gdl*N))
        K_global = K_global + K_global.T - sparse.diags(K_global.diagonal(), dtype='float')
        
        C_global = sparse.csr_matrix((data_C, (rows_c, cols_c)), shape=(gdl*N,gdl*N))
        C_global.eliminate_zeros() #eliminar os valores iguais a zero (no caso linear, a diagonal principal assume valores iguais a 0 em muitas posicoes)
        
        M_global = sparse.csr_matrix((data_M, (rows, cols)), shape=(gdl*N,gdl*N))
        M_global = M_global + M_global.T - sparse.diags(M_global.diagonal(), dtype='float')
        
        self.M = M_global
        self.C = C_global
        self.K = K_global

#        return M_global,C_global,K_global
    
    def solve_time_step(self,U):
        N = self.Nnodes
        gdl = self.gdl
        dt = self.dt
        
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

        
        A = sparse.csr_matrix((data_A, (rows_A, cols_A)), shape=(gdl*N,gdl*N))        
              
        #Construindo o vetor livre
        F = A.dot(U)
        A_sq_U = A.dot(F)
        
        f_global = dt*self.C.dot(F) -0.5*dt*dt*self.K.dot(A_sq_U)
        
        #Boundary conditions
#        w = 1e20
        for k in range(0,len(bc)):
            pos = bc[k].node.id
            
            if pos==0:
                f_global[pos*gdl:(pos+1)*gdl] = f_global[pos*gdl:(pos+1)*gdl] + dt*bc[k].value    
            elif pos==N-1:
                f_global[pos*gdl:(pos+1)*gdl] = f_global[pos*gdl:(pos+1)*gdl] - dt*bc[k].value    
            
#            self.M[pos*gdl,pos*gdl]     =  self.M[pos*gdl,pos*gdl] + w
#            self.M[pos*gdl+1,pos*gdl+1] =  self.M[pos*gdl+1,pos*gdl+1] + w
#            self.M[pos*gdl+2,pos*gdl+2] =  self.M[pos*gdl+2,pos*gdl+2] + w
#            
#            f_global[pos*gdl] = bc[k].value[0]*w
#            f_global[pos*gdl+1] = bc[k].value[1]*w
#            f_global[pos*gdl+2] = bc[k].value[2]*w
            
        return spsolve(self.M, f_global)

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
bc.append(cls_boundary_conditions(node[0]      ,   gdl, [0,  1,0], 'flux'))
bc.append(cls_boundary_conditions(node[N_final-1], gdl, [0,0.1,0], 'flux'))
#bc.append(cls_boundary_conditions(node[0]      ,   gdl, [0,0,0], 'value'))
#bc.append(cls_boundary_conditions(node[N_final-1], gdl, [0,0,0], 'value'))


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
problem.compute_global_matrices() #calculo das matrices globais pode ser feito apenas uma vez antes do loop principal


################################### LOOP PRINCIPAL  ###########################################
densities = U[0:N_final*gdl:gdl].copy() #Criando contourf plot da densidade em x,t

t = 0
time = []
time.append(t)

#tic = sys_time.clock()
while t<t_final:
        Delta_U = problem.solve_time_step(U)
        U += Delta_U
        t += delta_t
        
        densities = np.vstack([densities,U[0:N_final*gdl:gdl]]) #Criando contourf plot da densidade em x,t
        time.append(t)    
#print(sys_time.clock() - tic)
    
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
