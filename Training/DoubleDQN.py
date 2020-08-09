import numpy as np
from numpy.testing import assert_equal
from scipy.integrate import complex_ode
#from scipy.stats import kurtosis
from scipy.stats import moment

# time measurement
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d, Axes3D

import tensorflow as tf


"""Definition of the model parameter"""

def laser_simulation(uvt, alpha1, alpha2, alpha3, alphap,K):
    # parameters of the Maxwell equation
    
    D = -0.4
    E0 = 4.23
    tau = 0.1
    g0 = 1.73
    Gamma = 0.1
    
    
    Z = 1.5         # cavity length
    T = 60
    n = 256       # t slices
    Rnd = 500     # round trips
    t2 = np.linspace(-T/2,T/2,n+1)
    t_dis = t2[0:n].reshape([1,n])   # time discretization
    k = (2*np.pi/T)*np.concatenate((np.linspace(0,n//2-1,n//2),
                          np.linspace(-n//2,-1,n//2)),0)
    ts=[]
    ys=[]
    t0=0.0
    tend=1
    
    # waveplates & polarizer
    W4 = np.array([[np.exp(-1j*np.pi/4), 0],[0, np.exp(1j*np.pi/4)]]); # quarter waveplate
    W2 = np.array([[-1j, 0],[0, 1j]]);  # half waveplate
    WP = np.array([[1, 0], [0, 0]]);  # polarizer
    
    # waveplate settings
    R1 = np.array([[np.cos(alpha1), -np.sin(alpha1)], 
                   [np.sin(alpha1), np.cos(alpha1)]])
    R2 = np.array([[np.cos(alpha2), -np.sin(alpha2)], 
                   [np.sin(alpha2), np.cos(alpha2)]])
    R3 = np.array([[np.cos(alpha3), -np.sin(alpha3)], 
                   [np.sin(alpha3), np.cos(alpha3)]])
    RP = np.array([[np.cos(alphap), -np.sin(alphap)], 
                   [np.sin(alphap), np.cos(alphap)]])
    J1 = np.matmul(np.matmul(R1,W4),np.transpose(R1))
    J2 = np.matmul(np.matmul(R2,W4),np.transpose(R2))
    J3 = np.matmul(np.matmul(R3,W2),np.transpose(R3))
    JP = np.matmul(np.matmul(RP,WP),np.transpose(RP))
    
    # transfer function
    Transf = np.matmul(np.matmul(np.matmul(J1,JP),J2),J3)
    
    urnd=np.zeros([Rnd, n], dtype=complex)
    vrnd=np.zeros([Rnd, n], dtype=complex)
    t_dis=t_dis.reshape(n,)
    energy=np.zeros([1,Rnd])
    
    # definition of the rhs of the ode
    def mlock_CNLS_rhs(ts, uvt):
        [ut_rhs,vt_rhs] = np.split(uvt,2)
        u = np.fft.ifft(ut_rhs)
        v = np.fft.ifft(vt_rhs)
        # calculation of the energy function
        E = np.trapz(np.conj(u)*u+np.conj(v)*v,t_dis)
        
        # u of the rhs
        urhs = -1j*0.5*D*(k**2)*ut_rhs - 1j*K*ut_rhs +                 1j*np.fft.fft((np.conj(u)*u+ (2/3)*np.conj(v)*v)*u +                               (1/3)*(v**2)*np.conj(u)) +                 2*g0/(1+E/E0)*(1-tau*(k**2))*ut_rhs - Gamma*ut_rhs
        
        # v of the rhs
        vrhs = -1j*0.5*D*(k**2)*vt_rhs + 1j*K*vt_rhs +                 1j*np.fft.fft((np.conj(v)*v+(2/3)*np.conj(u)*u)*v +                               (1/3)*(u**2)*np.conj(v) ) +                 2*g0/(1+E/E0)*(1-tau*(k**2))*vt_rhs - Gamma*vt_rhs
         
        return np.concatenate((urhs, vrhs),axis=0)
    
    # definition of the solution output for the ode integration
    def solout(t,y):
        ts.append(t)
        ys.append(y.copy())
        
    start = time.time()
    
    uv_list = []
    norms = []
    change_norm = 100
    jrnd = 0
    # solving the ode for Rnd rounds
    while(jrnd < Rnd and change_norm > 1e-6):
        ts = []
        ys = []
        
        t0 = Z*jrnd
        tend = Z*(jrnd+1)
        
        uvtsol = complex_ode(mlock_CNLS_rhs)
        uvtsol.set_integrator(method='adams', name='dop853') # alternative 'dopri5'
        uvtsol.set_solout(solout)
        uvtsol.set_initial_value(uvt, t0)
        sol = uvtsol.integrate(tend)
        assert_equal(ts[0], t0)
        assert_equal(ts[-1], tend)
        
        u=np.fft.ifft(sol[0:n])
        v=np.fft.ifft(sol[n:2*n])
        
        urnd[jrnd,:]=u
        vrnd[jrnd,:]=v
        energy[0, jrnd]=np.trapz(np.abs(u)**2+np.abs(v)**2,t_dis)
        
        uvplus=np.matmul(Transf,np.transpose(np.concatenate((u.reshape(n,1),
                                                              v.reshape(n,1)),axis=1)))
        uv_list.append(np.concatenate((uvplus[0,:],
                                       uvplus[1,:]), axis=0))
        
        uvt=np.concatenate((np.fft.fft(uvplus[0,:]),
                                       np.fft.fft(uvplus[1,:])), axis=0)
        
        if jrnd > 0:
            phi=np.sqrt(np.abs(np.vstack(uv_list)[:,:n])**2 +                         np.abs(np.vstack(uv_list)[:,n:2*n])**2)
            change_norm=np.linalg.norm((phi[-1,:]-phi[len(phi)-2,:]))/             np.linalg.norm(phi[len(phi)-2,:])
            norms.append(change_norm) 
            
        jrnd += 1
    
    
    kur = np.abs(np.fft.fftshift(np.fft.fft(phi[-1,:])))
    #M4 = kurtosis(kur)
    M4 = moment(kur,4)/np.std(kur)**4
    
    end = time.time()
    # print(end-start)
    
    E = np.sqrt(np.trapz(phi[-1,:]**2, t_dis))
    
    states = np.array([E, M4, alpha1, alpha2, alpha3, alphap])
    
    """
    # surface plot 
    # create meshgrid
    X, Y = np.meshgrid(t_dis,np.arange(0,len(norms)))
    
    # figure urnd
    fig_urand = plt.figure()
    # ax = fig_urand.gca(projection='3d')
    ax = Axes3D(fig_urand)
    
    # plot the surface
    surf = ax.plot_surface(X, Y, np.abs(urnd[:len(norms),:]), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig_urand.colorbar(surf, shrink=0.5, aspect=5)
    
    
    # figure vrnd
    fig_vrand = plt.figure()
    # ax = fig_vrand.gca(projection='3d')
    ax = Axes3D(fig_vrand)
    
    # plot the surface
    surf = ax.plot_surface(X, Y, np.abs(vrnd[:len(norms),:]), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Add a color bar which maps values to colors.
    fig_vrand.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()
    """
    return (uvt,states)


def sech(x):
    # definition of the sech-function
    return np.cosh(x)**(-1)

def whitenoise(x):
    n = np.shape(x)[1]
    return 0.5*(np.random.random(n)+1j*np.random.random(n))


T = 60
n = 256
t2 = np.linspace(-T/2,T/2,n+1)
t_dis = t2[0:n].reshape([1,n])      # time discretization


class DoubleDQN(object):
    def __init__(self, n, n_actions, batch_size=8, double_dqn=False, learning_rate=0.002,
                 epsilon=1.0, epsilon_decay_rate=0.9998, gamma=0.99):
        self.n = n
        self.n_actions = n_actions
        self.double_dqn = double_dqn
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate
        self.gamma = gamma
        self.batch_size = batch_size

        self.learning_step = 0
        self.params_replace_step = 50
        self.tau = 0.01

        self.memory_size = 7000
        self.memory_counter = 0
        self.memory = np.zeros([self.memory_size, self.n*4*2+3*1+2])

        self._build_net()
        
        
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # self.update_params_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]  # update target net params
        self.update_params_op = [tf.assign(t, (1 - self.tau) * t + self.tau * e) for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self):

        k_init, b_init = tf.random_uniform_initializer(0., 0.2), tf.constant_initializer(0.1)

        # build eval net
        self.s = tf.placeholder(tf.float32, [None, 4*self.n+1], name='s')
        self.a = tf.placeholder(tf.int32, [None], name='a')
        self.s_ = tf.placeholder(tf.float32, [None, 4*self.n+1], name='s_')
        self.q_target = tf.placeholder(tf.float32, [None], name='q_target')
        
        
        h1_units = 256
        h2_units = 128
        h3_units = 64
        h4_units = 32
        

        with tf.variable_scope('eval_net'):
            
            self.s_expand = tf.expand_dims(self.s, -1)
            self.state = tf.concat([self.s_expand[:,:self.n,:], self.s_expand[:, self.n:self.n*2,:], 
                               self.s_expand[:, self.n*2:self.n*3,:],self.s_expand[:, self.n*3:-1,:]], axis=-1, name='concat_uv')
            self.state = tf.expand_dims(self.state, -1)
            self.conv_1 = tf.layers.conv2d(inputs=self.state, filters=16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='conv_1', trainable=False)
            self.max_1 = tf.layers.max_pooling2d(self.conv_1, (2,2), (2,2), padding='same', name='max_1')
            self.conv_2 = tf.layers.conv2d(inputs=self.max_1, filters=4, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='conv_2', trainable=False)
            self.max_2 = tf.layers.max_pooling2d(self.conv_2, (2,2), (2,2), padding='same', name='max_2')
            self.conv_3 = tf.layers.conv2d(inputs=self.max_2, filters=1, kernel_size=[2, 2], padding="same", activation=None, name='conv_3', trainable=False)
            
            self.state_dense = tf.abs(tf.squeeze(self.conv_3, [2, 3]))                               
            self.s_dense = tf.concat([self.state_dense, tf.reshape(self.s[:,-1], [-1, 1])], axis=-1, name='dense_s') 
            
            self.hidden_eval_1 = tf.layers.dense(inputs=self.s_dense, units=h1_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_eval_1')
            self.hidden_eval_2 = tf.layers.dense(inputs=self.hidden_eval_1, units=h2_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_eval_2')
            self.hidden_eval_3 = tf.layers.dense(inputs=self.hidden_eval_2, units=h3_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_eval_3')
            
            self.hidden_eval_4 = tf.layers.dense(inputs=self.hidden_eval_3, units=h4_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_eval_4')
            
            self.eval_net = tf.layers.dense(inputs=self.hidden_eval_4, units=self.n_actions, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                            bias_initializer=b_init, name='eval_net')

        # build target net
        with tf.variable_scope('target_net'):
            
            self.s_expand_ = tf.expand_dims(self.s_, -1)
            self.state_ = tf.concat([self.s_expand_[:,:self.n,:], self.s_expand_[:, self.n:self.n*2,:],
                                     self.s_expand_[:, self.n*2:self.n*3,:],self.s_expand_[:, self.n*3:-1,:]], axis=-1, name='concat_uv_')
            self.state_ = tf.expand_dims(self.state_, -1)
            self.conv_1_ = tf.layers.conv2d(inputs=self.state_, filters=16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='conv_1_', trainable=False)
            self.max_1_ = tf.layers.max_pooling2d(self.conv_1_, (2,2), (2,2), padding='same', name='max_1_')
            self.conv_2_ = tf.layers.conv2d(inputs=self.max_1_, filters=4, kernel_size=[3, 3], padding="same", activation=tf.nn.relu, name='conv_2_', trainable=False)
            self.max_2_ = tf.layers.max_pooling2d(self.conv_2_, (2,2), (2,2), padding='same', name='max_2_')
            self.conv_3_ = tf.layers.conv2d(inputs=self.max_2_, filters=1, kernel_size=[2, 2], padding="same", activation=None, name='conv_3_', trainable=False)
            
            self.state_dense_ = tf.abs(tf.squeeze(self.conv_3_, [2, 3]))
            self.s_dense_ = tf.concat([self.state_dense_, tf.reshape(self.s_[:,-1], [-1, 1])], axis=-1, name='dense_s')
                     
            self.hidden_tar_1 = tf.layers.dense(inputs=self.s_dense_, units=h1_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_tar_1')
            self.hidden_tar_2 = tf.layers.dense(inputs=self.hidden_tar_1, units=h2_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_tar_2')
            self.hidden_tar_3 = tf.layers.dense(inputs=self.hidden_tar_2, units=h3_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_tar_3')
            
            self.hidden_tar_4 = tf.layers.dense(inputs=self.hidden_tar_3, units=h4_units, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                           bias_initializer=b_init, name='hidden_tar_4')
            
            self.target_net = tf.layers.dense(inputs=self.hidden_tar_4, units=self.n_actions, activation=tf.nn.leaky_relu, kernel_initializer=k_init,
                                              bias_initializer=b_init, name='target_net')
            

        with tf.variable_scope('loss'):
            
            self.actions_reward = tf.multiply(self.eval_net, tf.one_hot(indices=self.a, depth=self.n_actions))
            self.actions_reward = tf.reduce_sum(self.actions_reward, reduction_indices=1)            
            self.loss = tf.reduce_mean(tf.square(self.q_target - self.actions_reward))
            print(self.loss.get_shape())
            
        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.train_op = self.optimizer.minimize(self.loss)
            
    

    def choose_action(self, sess, s):
        s = s[np.newaxis, :]
        if np.random.uniform() > self.epsilon:
            action_values = sess.run(self.eval_net, feed_dict={self.s: s})
            action = np.argmax(action_values)
        else:
            action = np.random.choice(range(self.n_actions))
        return action
    
    def infer_action(self, sess, s):
        s = s[np.newaxis, :]       
        action_values = sess.run(self.eval_net, feed_dict={self.s: s})
        action = np.argmax(action_values)       
        return action
    
    
    def store_transition(self, s, a, r, s_, terminal):
        transition = np.hstack((s, a, r, s_, terminal))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    

    def learn(self, sess):
        
        sess.run(self.update_params_op)

        if self.memory_counter <= self.memory_size:
            index = np.random.choice(self.memory_counter, self.batch_size)
        else:
            index = np.random.choice(self.memory_size, self.batch_size)
        batch_memory = self.memory[index, :]

        bs = batch_memory[:, :4*self.n+1]
        ba = batch_memory[:, 4*self.n+1]
        br = batch_memory[:, 4*self.n+2*1]
        bs_ = batch_memory[:, -(4*self.n+1+1):-1]
        bt = batch_memory[:, -1]

        q_target_next = sess.run(self.target_net, feed_dict={self.s_: bs_})

        if self.double_dqn:
            q_eval_next = sess.run(self.eval_net, feed_dict={self.s: bs_})
            a4next = np.argmax(q_eval_next, axis=1)
            q_target = br + self.gamma * q_target_next[np.arange(self.batch_size, dtype=np.int32), a4next]*bt
        else:
            q_next = np.max(q_target_next, axis=1)
            q_target = br + self.gamma*q_next*bt

        _, loss = sess.run([self.train_op, self.loss],
                            feed_dict={self.s: bs,
                                       self.a: ba,
                                       self.q_target: q_target})
              
        self.epsilon *= self.epsilon_decay_rate
        
        self.learning_step += 1
        if (self.learning_step%6000)==0:
            self.learning_rate *= 0.85
            
        return loss
    
        

EPISODES = 2000
batch_size = 8



tf.reset_default_graph()
agent = DoubleDQN(n=n, n_actions=3, batch_size=8, double_dqn=True, learning_rate=0.002, epsilon=1.0, epsilon_decay_rate=0.99997, gamma=0.99)



def ddqn_train(sess, K, alpha1_change, max_steps=50):
    plt.ion()
    total_r_infer = []
    total_alpha1_init = []
    reward_infer_max = 0.0
    
    with open('rewards.txt', 'w') as f:
        for episode in range(EPISODES):
        
            alpha1_init = np.random.uniform(-40*np.pi/180, -10*np.pi/180)
            total_alpha1_init.append(180*alpha1_init/np.pi)
            alpha1_cur = alpha1_init
            ep_step = 0
            ep_r = 0
            
            
            u=0.5*np.reshape(sech(t_dis/2), [n,])   # orthogonally polarized electric field
            v=0.5*np.reshape(sech(t_dis/2), [n,])   # envelopes in the optical fiber
            uv=np.concatenate([u, v], axis=0)
                       
            ut=np.fft.fft(u).reshape(n,)                         # fast fourier transformation of the
            vt=np.fft.fft(v).reshape(n,)                         # electrical fields
            uvt=np.concatenate([ut, vt], axis=0)                 # concatenation of ut and vt           
            
            s = np.concatenate((uv.real, uv.imag, np.array([alpha1_cur])), axis=0)      # birefringence K is not taken as part of state here (we assume constant K)
        
            cur_r = []
            cur_angle = []
            cur_alpha = []
            cur_loss = []
            
        
            while True:
                a = agent.choose_action(sess, s)
                # print(a)
                alpha1_cur += alpha1_change[a]
                cur_angle.append(a)
                
                (next_uvt, info) = laser_simulation(uvt, alpha1_cur, 3*np.pi/180, 28*np.pi/180, 88*np.pi/180, K)
                
                next_u = np.fft.ifft(next_uvt[:n])
                next_v = np.fft.ifft(next_uvt[n:])
                next_uv = np.concatenate([next_u, next_v], axis=0)
                s_ = np.concatenate((next_uv.real,next_uv.imag, np.array([alpha1_cur])), axis=0)
                r = info[0]/info[1]-0.1
                
                
                terminal = 1
                
                agent.store_transition(s, a, r, s_, terminal)
                
                r += 0.1
                cur_r.append(r)
                cur_alpha.append(180*alpha1_cur/np.pi)

                ep_r += r
                ep_step += 1

                if agent.memory_counter >= agent.batch_size:
                    loss = agent.learn(sess)
                    cur_loss.append(loss)                  

                
                if ep_step >= max_steps:
                    break
                    
                
                del s
                del uvt

                s = s_
                uvt = next_uvt              

                       
            print(cur_r)
            print(cur_alpha)
            print(cur_loss)
            print(cur_angle)
            print('Episode %d Reward %s' % (episode, ep_r))
            
            f.write("%s\n%s\n%s\n%s\n" % (cur_r,cur_alpha,cur_loss,cur_angle))
            f.write('Episode %d Reward %s\n' % (episode, ep_r))
        
            if (episode+1) % 10 == 0:
                saver.save(sess, './ddqn_weights')
                reward_infer = ddqn_infer(sess, K=0.1, alpha1_init=-15*np.pi/180, alpha1_change=[-np.pi/90, 0, np.pi/90])
                total_r_infer.append(reward_infer)
                
                if reward_infer>=reward_infer_max:
                    saver.save(sess, '/ddqn_weights_best')
                    reward_infer_max = reward_infer

                plt.cla()
                plt.plot(total_r_infer)
                plt.pause(0.0001)
        
        f.write("%s\n%s\n" % (total_alpha1_init,total_r_infer))
        
        plt.ioff()
        plt.show()
        
        
def ddqn_infer(sess, K, alpha1_init, alpha1_change, max_steps=20):
    alpha1_cur = alpha1_init

    u=0.5*np.reshape(sech(t_dis/2), [n,])   # orthogonally polarized electric field
    v=0.5*np.reshape(sech(t_dis/2), [n,])   # envelopes in the optical fiber
    uv=np.concatenate([u, v], axis=0)

    ut=np.fft.fft(u).reshape(n,)                         # fast fourier transformation of the
    vt=np.fft.fft(v).reshape(n,)                         # electrical fields
    uvt=np.concatenate([ut, vt], axis=0)                 # concatenation of ut and vt
    s = np.concatenate((uv.real, uv.imag, np.array([1.0*alpha1_cur])), axis=0)      # birefringence K is not taken as part of state here (we assume constant K)
    
    rewards = []
    ep_step = 0

    while True:
        a = agent.infer_action(sess, s)
        alpha1_cur += alpha1_change[a]
    
        (next_uvt, info) = laser_simulation(uvt, alpha1_cur, 3*np.pi/180, 28*np.pi/180, 88*np.pi/180, K)
        next_u = np.fft.ifft(next_uvt[:n])
        next_v = np.fft.ifft(next_uvt[n:])
        next_uv = np.concatenate([next_u, next_v], axis=0)
        s_ = np.concatenate((next_uv.real,next_uv.imag, np.array([1.0*alpha1_cur])), axis=0)
        r = info[0]/info[1]
    
        if ep_step >= max_steps:
            break
    
        s = s_
        uvt = next_uvt
        ep_step += 1
        rewards.append(r)

    return np.sum(rewards)



with tf.Session() as sess:
    saver = tf.train.Saver()
    ddqn_train(sess, K=0.1, alpha1_change=[-np.pi/90, 0, np.pi/90], max_steps=40)
